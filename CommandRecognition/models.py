import torch
from torch import nn
import torch.nn.functional as F


class SA_Layer(nn.Module):
    """ for 1 dim sequences """

    def __init__(self, elem_size, attention_hidden_dim=64, out_dim=64):
        super().__init__()
        self.query_lin = nn.Linear(elem_size, attention_hidden_dim)
        self.key_lin = nn.Linear(elem_size, attention_hidden_dim)
        self.value_lin = nn.Linear(elem_size, out_dim)
        self.softmax = nn.Softmax(dim=1)
        self.residual_conn_param = nn.Parameter(torch.rand(1) / 3)

    def forward(self, x):
        N, seq_size, elem_size = x.shape
        seq_queries = self.query_lin(x)
        seq_keys = self.key_lin(x)
        out = torch.empty(N, seq_size, 64)  # empty to save grads
        for i in range(N):
            curr_q = seq_queries[i]
            curr_k = seq_keys[i]
            curr_v = self.value_lin(x[i])
            attention_mat = torch.bmm(curr_q, torch.transpose(curr_k, 1, 0))
            attention = self.softmax(attention_mat, dim=1)  # softmax for each row
            final_rep = torch.bmm(attention, torch.transpose(curr_v, 1, 0))
            out[i] = final_rep  # + self.residual_conn_param * proj_values
        return out


class MultiHead_SA_Layer(nn.Module):
    def __init__(self, N, elem_size, attention_hidden_dim=64, out_dim=64):
        """
        :param N: number of heads
        """
        super().__init__()
        self.N = N
        self.out_dim = out_dim

        self.SAs = nn.ModuleList()  # otherwise it doesn't save grad... (as same for non empty tensors)
        for _ in range(N):
            self.SAs.append(SA_Layer(elem_size, attention_hidden_dim, out_dim))

    def forward(self, x):
        SAs_out = torch.empty(self.N, x.shape[0], self.out_dim)
        for i in range(self.N):
            SAs_out[i] = self.SAs[i](x)

        out = torch.sum(SAs_out, dim=0)
        return out


class CommandRecognitionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.simple_net = nn.Sequential(
            nn.Conv1d(1, 10, 200, stride=50),  # 16k -> 317
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Conv1d(10, 100, 30, stride=10),  # 317 -> 29
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Conv1d(100, 200, 29, stride=1),  # 29 -> 1
            nn.BatchNorm1d(200),
        )
        self.proj = nn.Linear(200, 35)

    def forward(self, x):
        out = self.simple_net(x).view(x.shape[0], -1)
        out = self.proj(out)
        return out  # I don't apply softmax as CE loss in pytorch adds it in train only
