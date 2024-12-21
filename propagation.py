import math
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, input):
        input_coal = input.coalesce()
        drop_val = F.dropout(input_coal._values(), self.p, self.training)
        return torch.sparse.FloatTensor(input_coal._indices(), drop_val, input.shape)


class MixedDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.dense_dropout = nn.Dropout(p)
        self.sparse_dropout = SparseDropout(p)

    def forward(self, input):
        if input.is_sparse:
            return self.sparse_dropout(input)
        else:
            return self.dense_dropout(input)


class MixedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Our fan_in is interpreted by PyTorch as fan_out (swapped dimensions)
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', a=math.sqrt(5))
        if self.bias is not None:
            _, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_out)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.bias is None:
            if input.is_sparse:
                res = torch.sparse.mm(input, self.weight)
            else:
                res = input.matmul(self.weight)
        else:
            if input.is_sparse:
                res = torch.sparse.addmm(self.bias.expand(input.shape[0], -1), input, self.weight)
            else:
                res = torch.addmm(self.bias, input, self.weight)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)


def sparse_matrix_to_torch(X):
    coo = X.tocoo()
    indices = np.array([coo.row, coo.col])
    return torch.sparse.FloatTensor(
        torch.LongTensor(indices),
        torch.FloatTensor(coo.data),
        coo.shape)


def calc_A_hat(adj_matrix: sp.spmatrix) -> sp.spmatrix:
    nnodes = adj_matrix.shape[0]
    A = adj_matrix + sp.eye(nnodes)
    D_vec = np.sum(A, axis=1)
    if isinstance(A, sp.spmatrix):
        D_vec = D_vec.A1
    D_vec_invsqrt_corr = 1 / np.sqrt(D_vec)
    D_invsqrt_corr = sp.diags(D_vec_invsqrt_corr)
    return D_invsqrt_corr @ A @ D_invsqrt_corr


def calc_ppr_exact(adj_matrix: sp.spmatrix, alpha: float) -> np.ndarray:
    nnodes = adj_matrix.shape[0]
    M = calc_A_hat(adj_matrix)
    A_inner = sp.eye(nnodes) - (1 - alpha) * M
    return alpha * np.linalg.inv(A_inner.toarray())


class PPRExact(nn.Module):
    def __init__(self, adj_matrix: sp.spmatrix, alpha: float, drop_prob: float = None):
        super().__init__()

        ppr_mat = calc_ppr_exact(adj_matrix, alpha)
        self.register_buffer('mat', torch.FloatTensor(ppr_mat))

        if drop_prob is None or drop_prob == 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(drop_prob)

    def forward(self, predictions: torch.FloatTensor, idx: torch.LongTensor):
        return self.dropout(self.mat[idx]) @ predictions


class PPRPowerIteration(nn.Module):
    def __init__(self, adj_matrix: sp.spmatrix, alpha: float, niter: int, drop_prob: float = None):
        super().__init__()
        self.alpha = alpha
        self.niter = niter

        M = calc_A_hat(adj_matrix)
        self.register_buffer('A_hat', sparse_matrix_to_torch((1 - alpha) * M))

        if drop_prob is None or drop_prob == 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(drop_prob)

    def forward(self, local_preds: torch.FloatTensor):
        preds = local_preds
        for _ in range(self.niter):
            A_drop = self.dropout(self.A_hat)
            preds = A_drop.cuda() @ preds + self.alpha * local_preds
        return preds
