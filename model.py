import torch
from torch import nn
from propagation import PPRPowerIteration, MixedLinear, MixedDropout


class Encoder(nn.Module):
    r""" Encoder

    Description
    -----------
    Encoder from `Predict then Propagate: Graph Neural Networks meet Personalized PageRank https://arxiv.org/abs/1810.05997>`__.
    GNN encoder f_h to generate node embeddings for each graph.

    Parameters
    ----------
    in_dim : int
        Input dimension of features.
    num_hidden : int
        Number of hidden units.
    feat_drop : float, optional
        Dropout rate.
    """

    def __init__(self, in_dim, num_hidden, feat_drop):
        super(Encoder, self).__init__()
        hiddenunits = [num_hidden]
        fcs = [MixedLinear(in_dim, hiddenunits[0], bias=False)]
        for i in range(1, len(hiddenunits)):
            fcs.append(nn.Linear(hiddenunits[i - 1], hiddenunits[i], bias=False))
        self.fcs = nn.ModuleList(fcs)
        self.reg_params = list(self.fcs[0].parameters())
        if feat_drop == 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(feat_drop)
        self.act_fn = nn.ReLU()

    def forward(self, attr_matrix: torch.sparse.FloatTensor):
        layer_inner = self.act_fn(self.fcs[0](self.dropout(attr_matrix)))
        for fc in self.fcs[1:]:
            layer_inner = self.act_fn(fc(layer_inner))
        return layer_inner


class NodeClassifier(nn.Module):
    r""" Domain-specific Node Classifiers

    Description
    -----------
    multiple unshared-weight domain-specific node classifiers {f_y^k}_(k=1)^K are added to adapt to various data
    distributions of different source graphs.

    Parameters
    ----------
    num_hidden : int
        The number of hidden units in each layer.
    num_classes : int
        The number of classes.
    feat_drop : float
        The dropout rate on features.
    num_source : int
        The number of source networks.
    """

    def __init__(self, num_hidden, num_classes, feat_drop, num_source):
        super(NodeClassifier, self).__init__()
        self.num_source = num_source
        self.node_classifier_list = nn.ModuleList()
        for i in range(num_source):
            self.node_classifier_list.append(nn.Linear(num_hidden, num_classes, bias=False))
        self.dropout = MixedDropout(feat_drop)

    def forward(self, encoded_output_list, A_list, step, is_target=False):
        logits_list = []
        if is_target:
            for i in range(self.num_source):
                local_logits = self.node_classifier_list[i](self.dropout(encoded_output_list))
                propagation = PPRPowerIteration(A_list, alpha=0.1, niter=step)
                final_logits = propagation(local_logits)
                logits_list.append(final_logits)
        else:
            for i in range(self.num_source):
                local_logits = self.node_classifier_list[i](self.dropout(encoded_output_list[i]))
                propagation = PPRPowerIteration(A_list[i], alpha=0.1, niter=step)
                final_logits = propagation(local_logits)
                logits_list.append(final_logits)
        return logits_list


class PMGCDA(nn.Module):
    r""" Prototype-based Multi-source Graph Contrastive Domain Adaptation

    Description:
    ----------
    The model architecture of PMGCDA, which contains a shared-weight GNN encoder along
    with multiple unshared-weight domain-specific node classifiers,
     a transferability weight learning module (domain level and node level),
     a prototype-based graph contrastive domain adaptation module,
     and a pseudo label learning module.

    Parameters:
    ----------
    in_dim : int
        Input dimension.
    num_hidden : int
        Hidden dimension.
    feat_drop : float
        Dropout rate on features.
    num_classes : int
        Number of classes.
    num_source : int
        Number of source domains.
    """

    def __init__(self, in_dim, num_hidden, feat_drop, num_classes, num_source):
        super(PMGCDA, self).__init__()
        self.network_embedding = Encoder(in_dim, num_hidden, feat_drop)
        self.node_classifier_list = NodeClassifier(num_hidden, num_classes, feat_drop, num_source)

    def forward(self, num_source, features_s_list, features_t, A_s_list, A_t, step):
        h_s_list = [self.network_embedding(features_s_list[i]) for i in range(num_source)]
        pred_logit_s_list = self.node_classifier_list(h_s_list, A_s_list, step)
        emb_s_list = [h_s_list[i].reshape(h_s_list[i].shape[0], -1) for i in range(num_source)]

        h_t = self.network_embedding(features_t)
        pred_logit_t_list = self.node_classifier_list(h_t, A_t, step, is_target=True)
        emb_t = h_t.reshape(h_t.shape[0], -1)

        return pred_logit_s_list, pred_logit_t_list, emb_s_list, emb_t