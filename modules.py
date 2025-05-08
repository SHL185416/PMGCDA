import numpy as np
import torch
import torch.nn.functional as F

from utils import *


def domain_transfer_weight(pred_logit_t_list, num_source):
    r"""domain transferability weights based on information entropy.

    :arg
        pred_logit_t_list (list): predicted logits from target domain.
        num_source (int): number of source domains.
    :return
        w_k_list (list): domain transferability weights for each source domain.
    """
    w_k_list = [0] * num_source
    for i in range(num_source):
        entropy = (-F.softmax(pred_logit_t_list[i].detach(), dim=1) * torch.log_softmax(
            pred_logit_t_list[i].detach(),
            dim=1)).sum(1)
        w_k_list[i] = 1 / entropy.mean()
    w_k_sum = sum(w_k_list)
    for i in range(num_source):
        w_k_list[i] = w_k_list[i] / w_k_sum
    return w_k_list


def prediction_source_confidence(pred_t_list, num_source, pred_label_t):
    _, labels = torch.max(pred_label_t, dim=1)

    source_confidence_list = []
    for i in range(num_source):
        prototype = calculate_prototype(pred_label_t, pred_t_list[i])
        source_confidence = []
        for j in range(len(pred_t_list[i])):
            if (prototype[labels[j]].sum() == 0):
                source_confidence.append(torch.tensor(0.0).to(pred_label_t.device))
            else:
                similarity = -1 * measures_similarity(pred_t_list[i][j], prototype[labels[j]])
                source_confidence.append(similarity)
        source_confidence = torch.stack(source_confidence)
        source_confidence_list.append(source_confidence)
    return source_confidence_list


def prediction_source_diversity(pred_t_list, num_source, pred_label_t):
    _, labels = torch.max(pred_label_t, dim=1)
    source_diversity_list = []
    for i in range(num_source):
        prototype = calculate_prototype(pred_label_t, pred_t_list[i])
        mean_prototype = prototype.mean(dim=0)
        source_diversity = []
        diversity = []
        #  prediction diversity between prototype and mean_prototype
        for c in range(len(prototype)):
            if prototype[c].sum() == 0:
                diversity.append(torch.tensor(0.0).to(pred_label_t.device))
            else:
                diversity.append(measures_similarity(prototype[c], mean_prototype))
        # Assign the corresponding diversity based on the label of the target node
        for j in range(len(pred_t_list[i])):
            label = labels[j]
            source_diversity.append(diversity[label])
        source_diversity = torch.stack(source_diversity)
        source_diversity_list.append(source_diversity)
    return source_diversity_list


def prediction_source_similarity(pred_t_list, num_source):
    source_similarity_list = []
    pred_t_mean = pred_t_list[0]
    for i in range(num_source - 1):
        pred_t_mean = pred_t_mean + pred_t_list[i + 1]
    pred_t_mean = pred_t_mean / num_source
    for i in range(num_source):
        source_similarity = []
        for j in range(len(pred_t_list[i])):
            similarity = -1 * measures_similarity(pred_t_list[i][j], pred_t_mean)
            source_similarity.append(similarity)
        source_similarity = torch.stack(source_similarity)
        source_similarity_list.append(source_similarity)
    return source_similarity_list


def node_transfer_weight(pred_logit_t_list, pred_label_t, num_source):
    r"""node transferability weights

    :arg
        pred_logit_t_list: the predicted logits of each domain specific node classifier for the target node.
        pred_label_t: the predicted label of the target node.
        num_source: the number of source domains.
    :return
        node transferability weights.
    """
    logit_t_list = []
    for i in range(num_source):
        logit_t_list.append(F.softmax(pred_logit_t_list[i].detach(), dim=1))

    confidences = prediction_source_confidence(logit_t_list, num_source, pred_label_t)
    diversities = prediction_source_diversity(logit_t_list, num_source, pred_label_t)
    similarities = prediction_source_similarity(logit_t_list, num_source)

    importances = []
    for i in range(num_source):
        importances.append(confidences[i] + similarities[i] + diversities[i] + 1e-8)
    return F.softmax(torch.stack(importances), dim=0)


def pseudo_labeling(pred_logit_t_list, tau_p, domain_w_k_list, node_w_k_list, args):
    r"""pseudo-labeling learning.

    :arg
        pred_logit_t_list (list): predicted logits from target domain.
        tau_p (float): threshold for pseudo-labeling learning.
        w_k_list (list): transferability weights for each source domain.
    :return
        loss: pseudo labeling loss.
    """
    indices_list = [0] * args.num_source
    pred_label_clf_list = [0] * args.num_source
    for j in range(args.num_source):
        _, indices_list[j] = torch.max(pred_logit_t_list[j], dim=1)
        pred_label_clf_list[j] = one_hot_encode_torch(indices_list[j], pred_logit_t_list[j].shape[1]).to(args.device)

    pred_label_pl = calculate_pred_label_t(pred_label_clf_list).float()
    pred_label_pl = pred_label_pl.to(indices_list[0].device)
    loss = 0
    pred_logit_node_t_softmax_list = [0] * args.num_source
    for j in range(args.num_source):
        pred_logit_node_t_softmax_list[j] = F.softmax(pred_logit_t_list[j], dim=1)
        pred_logit_node_t_softmax_list[j] = pred_logit_node_t_softmax_list[j].to(indices_list[0].device)

    for i in range(0, pred_logit_t_list[0].shape[1]):
        positive_idx_list = [0] * args.num_source
        for j in range(args.num_source):
            positive_idx_list[j] = np.array(
                torch.where((pred_logit_node_t_softmax_list[j][:, i] >= tau_p) * (pred_label_pl[:, i] > 0.0))[0].cpu())
        positive_idx = positive_idx_list[0]
        for k in range(args.num_source):
            positive_idx = np.intersect1d(positive_idx, positive_idx_list[k])
        for j in range(args.num_source):
            if positive_idx.size > 0:
                ce_loss = F.cross_entropy(
                    pred_logit_t_list[j][positive_idx],
                    indices_list[j][positive_idx],
                    reduction='none')
                loss += domain_w_k_list[j] * (ce_loss * node_w_k_list[j][positive_idx]).mean()
    return loss


def prototype_loss_st(local_prototype_s, local_prototype_t, update_count_s, domain_w_k_list, args):
    r"""prototypical graph contrastive domain adaptation loss.

    :arg
        local_prototype_s: local prototypes of source graphs.
        local_prototype_t: local prototypes of target graph.
        update_count_s: the update count of source graphs.
        domain_w_k_list: the transferability weight of each source domain.
        args: the arguments of the model.
    :return
        loss: prototypical graph contrastive domain adaptation loss.
    """
    local_prototype_loss = 0
    local_prototype = local_prototype_s
    local_prototype.append(local_prototype_t[0])
    local_prototype.append(local_prototype_t[1])
    for i in range(args.num_class):
        for j in range(args.num_source):
            for k in range(j + 1, args.num_source * 2):
                if k >= args.num_source:
                    local_prototype_loss = local_prototype_loss + domain_w_k_list[j] * torch.cosine_similarity(
                        local_prototype[j][i],
                        local_prototype[k][i],
                        dim=0)
                else:
                    local_prototype_loss = local_prototype_loss + torch.cosine_similarity(local_prototype[j][i],
                                                                                          local_prototype[k][i],
                                                                                          dim=0)
    global_prototype = 0
    update_sum = 0
    for i in range(args.num_source):
        temp = domain_w_k_list[i] * local_prototype_s[i] * update_count_s[i].unsqueeze(1)
        update_sum = update_sum + update_count_s[i].unsqueeze(1)
        global_prototype = global_prototype + temp
    global_prototype = global_prototype / update_sum

    global_prototype_loss = 0
    for i in range(args.num_class - 1):
        for j in range(i + 1, args.num_class):
            global_prototype_loss = global_prototype_loss + torch.cosine_similarity(global_prototype[i],
                                                                                    global_prototype[j], dim=0)
            for k in range(args.num_source):
                global_prototype_loss = global_prototype_loss + domain_w_k_list[k] * \
                                        torch.cosine_similarity(global_prototype[i], local_prototype_t[k][j], dim=0)

    return global_prototype_loss - local_prototype_loss
