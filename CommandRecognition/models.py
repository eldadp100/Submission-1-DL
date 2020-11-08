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
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)
