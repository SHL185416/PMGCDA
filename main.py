import argparse
import random
import dgl
import torch

from torch import nn
from metrics import f1_scores
from model import PMGCDA
from modules import *
from utils import *

# 0. Set model parameters
parser = argparse.ArgumentParser()
# 0.1 Equipment and number of iterations settings
parser.add_argument("--gpu", type=int, default=0, help="which GPU to use. Set -1 to use CPU.")
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument("--batch_size", type=int, default=8000, help="batch_size for each domain")
# 0.2 Optimizer settings
parser.add_argument('--lr_ini', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--l2_w', type=float, default=0.01, help='weight of L2-norm regularization')
# 0.3 GAT settings
parser.add_argument("--num_hidden", type=int, default=64, help="number of hidden units")
parser.add_argument("--step", type=int, default=10, help="The propagation layers of APPNP")
parser.add_argument("--in_drop", type=float, default=0.4, help="input feature dropout")
parser.add_argument("--random_number", type=int, default=1, help="random seed")
# 0.5 Pseudo label learning settings
parser.add_argument("--tau_p", type=float, default=0.5, help="tau_p for pseudo label learning")
# 0.6 The trade-off parameters
parser.add_argument("--Clf_wei", type=float, default=1, help="weight of clf loss")
parser.add_argument("--P_wei", type=float, default=1, help="weight of pseudo label learning loss")
parser.add_argument("--Prot_wei", type=float, default=0.01, help="weight of Psd_wei loss")
# 0.7 Dataset settings
parser.add_argument('--target', type=str, default='citation1_citationv1', help='target dataset name')
parser.add_argument('--data_key', type=str, default='citation', help='dataset key')
args = parser.parse_args()
# 1. Set device and file path
if args.gpu != -1 and torch.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'
f = open(f'./output/{args.data_key}/otherSet_{args.target}.txt', 'a')
f.write('{}\n'.format(args))
f.flush()
source_list = ["citation1_acmv9", "citation1_citationv1", "citation1_dblpv7"]
if args.target not in source_list:
    source_list = ["citation2_acmv8", "citation2_citationv1", "citation2_dblpv4"]
source_list.remove(args.target)
num_source = len(source_list)
args.num_source = num_source
A_s_list = [0] * num_source
X_s_list = [0] * num_source
Y_s_list = [0] * num_source
num_nodes_s_list = [0] * num_source
# 2. Load dataset
"""2.1 Load data from the source network"""
for i in range(num_source):
    A_s_list[i], X_s_list[i], Y_s_list[i] = load_citation(f"./input/{args.data_key}/{source_list[i]}.mat")
    num_nodes_s_list[i] = X_s_list[i].shape[0]
num_feat = X_s_list[0].shape[1]
num_class = Y_s_list[0].shape[1]
args.num_class = num_class
"""2.2 Load data from the target network"""
A_t, X_t, Y_t = load_citation(f"./input/{args.data_key}/{args.target}.mat")
num_nodes_t = X_t.shape[0]
features_s_list = [torch.Tensor(X_s_list[i].todense()).to(args.device) for i in range(num_source)]
features_t = torch.Tensor(X_t.todense()).to(args.device)
# 3. Definitions of model variables

ST_max = max([X_s_list[i].shape[0] for i in range(0, num_source)])
ST_max = max(ST_max, X_t.shape[0])
microAllRandom = []
macroAllRandom = []
best_microAllRandom = []
best_macroAllRandom = []
numRandom = args.random_number + 5
tau_p = args.tau_p

for random_state in range(args.random_number, numRandom):
    print('%d-th random split' % random_state)
    # 4. Set the nonce seed, model and loss
    random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state) if torch.cuda.is_available() else None
    np.random.seed(random_state)
    model = PMGCDA(
        in_dim=num_feat,
        num_hidden=args.num_hidden,
        feat_drop=args.in_drop,
        num_classes=num_class,
        num_source=num_source
    )
    model = model.to(args.device)
    clf_loss_f = nn.CrossEntropyLoss()
    best_epoch = 0
    best_micro_f1 = 0
    best_macro_f1 = 0
    pred_label = torch.zeros(Y_t.shape).to(args.device)
    update_count_s = torch.zeros(num_source, num_class).to(args.device)
    update_count_t = torch.zeros(num_source, num_class).to(args.device)
    emb_dim = args.num_hidden
    prototype_s_list_late = torch.zeros(num_source, num_class, num_class).to(args.device)
    prototype_t_list_late = torch.zeros(num_source, num_class, num_class).to(args.device)
    for epoch in range(1, args.epochs + 1):
        # 5. Use random sampling to sample from a dataset
        batch_s_list = [0] * num_source
        args_list = [mini_batch(X_s_list[i], Y_s_list[i], A_s_list[i], ST_max, args.batch_size) for i in
                     range(num_source)]
        args_list.append(mini_batch(X_t, pred_label, A_t, ST_max, args.batch_size))
        for batch_idx, batch_data in enumerate(zip(*args_list)):
            batch_s_list = [data[:4] for data in batch_data[:-1]]
            batch_t = batch_data[-1]
            feat_s_list = [0] * num_source
            label_s_list = [0] * num_source
            adj_s_list = [0] * num_source
            shuffle_index_s_list = [0] * num_source
            for i in range(num_source):
                feat_s_list[i], label_s_list[i], adj_s_list[i], shuffle_index_s_list[i] = batch_s_list[i]
                feat_s_list[i] = torch.FloatTensor(feat_s_list[i].toarray()).to(args.device)
                label_s_list[i] = torch.FloatTensor(label_s_list[i]).to(args.device)
            feat_t, pred_label_t, adj_t, shuffle_index_t = batch_t
            feat_t = torch.FloatTensor(feat_t.toarray()).to(args.device)

            p = float(epoch) / args.epochs
            lr = args.lr_ini / (1. + 10 * p) ** 0.75
            # 6. Train the model
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=args.l2_w)
            optimizer.zero_grad()
            pred_logit_s_list, pred_logit_t_list, emb_s_list, emb_t = model(num_source,
                                                                            feat_s_list,
                                                                            feat_t,
                                                                            adj_s_list,
                                                                            adj_t, step=args.step)
            """6.1 domain transferability weights and node transferability weights"""
            domain_weight_list = domain_transfer_weight(pred_logit_t_list, num_source)
            node_weight_list = node_transfer_weight(pred_logit_t_list, pred_label_t, num_source)
            pred_logit_t_withNW = torch.zeros_like(pred_logit_t_list[0]).to(args.device)
            for i in range(num_source):
                pred_logit_t_withNW = pred_logit_t_withNW + \
                                      pred_logit_t_list[i].detach() * node_weight_list[i].unsqueeze(1)
            """6.2 node classification loss"""
            clf_loss_list = [0] * num_source
            for i in range(num_source):
                clf_loss_list[i] = domain_weight_list[i] * clf_loss_f(pred_logit_s_list[i],
                                                                   torch.argmax(label_s_list[i], 1))
            """6.3 pseudo label learning"""
            p_loss = pseudo_labeling_pl(pred_logit_t_list, tau_p, domain_weight_list, args)
            print("p_loss:", p_loss)
            """6.4 prototype-based graph contrastive domain adaptation"""
            prototype_s_list = []
            prototype_t_list = []
            for i in range(num_source):
                pred_prob_xs = torch.softmax(pred_logit_s_list[i], dim=1)
                pred_prob_xt = torch.softmax(pred_logit_t_list[i], dim=1)
                prototype_s = calculate_prototype_withCount(label_s_list[i], pred_prob_xs, update_count_s[i],
                                                            prototype_s_list_late[i])
                prototype_t = calculate_prototype_withCount(pred_label_t, pred_prob_xt, update_count_t[i],
                                                            prototype_t_list_late[i])
                prototype_s_list.append(prototype_s)
                prototype_t_list.append(prototype_t)
                prototype_s_list_late[i] = prototype_s.detach()
                prototype_t_list_late[i] = prototype_t.detach()
            pro_loss = prototype_loss_st(prototype_s_list, prototype_t_list, update_count_s, domain_weight_list, args)
            print("pro_loss:", pro_loss)
            for i in range(num_source):
                prototype_s_list_late[i] = prototype_s.detach()
            total_loss = args.P_wei * p_loss + \
                         args.Clf_wei * sum(clf_loss_list) + \
                         args.Prot_wei * pro_loss
            total_loss.backward()
            optimizer.step()
        # 7. Compute evaluation on test data by the end of each epoch
        model.eval()
        with torch.no_grad():
            pred_logit_s_list, pred_logit_t_list, emb_s_list, emb_t = model(num_source,
                                                                            features_s_list,
                                                                            features_t,
                                                                            A_s_list,
                                                                            A_t, step=args.step)
            pred_label_clf_list = []
            for i in range(num_source):
                _, indices = torch.max(pred_logit_t_list[i], dim=1)
                pred_label_clf_list.append(one_hot_encode_torch(indices, num_class))
                print("accuracy of clf%d label" % i, f1_scores(pred_label_clf_list[i].cpu(), Y_t))
            pred_label = calculate_pred_label_t(pred_label_clf_list).float()
            if epoch % 5 == 0:
                print(f"{epoch + 1} pseudo_label {pred_label.sum() / pred_label.shape[0]} %")
            if torch.any(pred_label != 0, dim=1).sum() > 0:
                print("accuracy of refined label by both clustering and clf",
                      f1_scores(pred_label[torch.any(pred_label != 0, dim=1)].cpu(),
                                Y_t[torch.any(pred_label != 0, dim=1).cpu()]))
            """7.1 Domain transferability weights and node transferability weights"""
            domain_weight_list = domain_transfer_weight(pred_logit_t_list, num_source)
            node_weight_list = node_transfer_weight(pred_logit_t_list, pred_label.to(args.device), num_source)
            pred_logit_t_withNW = torch.zeros_like(pred_logit_t_list[0]).to(args.device)
            for i in range(num_source):
                pred_logit_t_withNW = pred_logit_t_withNW + \
                                      pred_logit_t_list[i].detach() * node_weight_list[i].unsqueeze(1)
            """7.2 Calculates the probabilities for the source and target domains"""
            pred_prob_xt = torch.softmax(pred_logit_t_withNW,dim=1)
            pred_prob_xs_list = []
            for i in range(num_source):
                pred_prob_xs_list.append(torch.softmax(pred_logit_s_list[i],dim=1))
            """7.3 Calculate s-t f1_scores"""
            for i in range(num_source):
                f1_s = f1_scores(pred_prob_xs_list[i].cpu(), Y_s_list[i])
                print('epoch %d: Source%d micro-F1: %f, macro-F1: %f' % (epoch, i, f1_s[0], f1_s[1]))
            f1_t = f1_scores(pred_prob_xt.cpu(), Y_t)
            print('epoch %d: Target testing micro-F1: %f, macro-F1: %f' % (epoch, f1_t[0], f1_t[1]))
            if f1_t[1] > best_macro_f1:
                best_micro_f1 = f1_t[0]
                best_macro_f1 = f1_t[1]
                best_epoch = epoch
    print('Target best epoch %d, micro-F1: %f, macro-F1: %f' % (best_epoch, best_micro_f1, best_macro_f1))
    microAllRandom.append(float(f1_t[0]))
    macroAllRandom.append(float(f1_t[1]))
    best_microAllRandom.append(float(best_micro_f1))
    best_macroAllRandom.append(float(best_macro_f1))
# 8. Record the results
micro = np.mean(microAllRandom)
macro = np.mean(macroAllRandom)
micro_sd = np.std(microAllRandom)
macro_sd = np.std(macroAllRandom)
print("The avergae micro and macro F1 scores over {} random initializations are:  {} +/- {} and {} +/- {}: ".format(
    numRandom - 1, micro, micro_sd, macro, macro_sd))
f.write("The avergae micro and macro F1 scores over {} random initializations are:  {} +/- {} and {} +/- {}: \n".format(
    numRandom - 1, micro, micro_sd, macro, macro_sd))

best_micro = np.mean(best_microAllRandom)
best_macro = np.mean(best_macroAllRandom)
best_micro_sd = np.std(best_microAllRandom)
best_macro_sd = np.std(best_macroAllRandom)
print(
    "The avergae best micro and macro F1 scores over {} random initializations are:  {} +/- {} and {} +/- {}: ".format(
        numRandom - 1, best_micro, best_micro_sd, best_macro, best_macro_sd))
f.write(
    "The avergae best micro and macro F1 scores over {} random initializations are:  {} +/- {} and {} +/- {}: \n".format(
        numRandom - 1, best_micro, best_micro_sd, best_macro, best_macro_sd))
f.flush()
f.close()
