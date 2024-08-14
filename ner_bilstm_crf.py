import torch
import torch.nn as nn

class LSTMblock(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.linear_ih = nn.Linear(input_size, 4 * hidden_size)
        self.linear_hh = nn.Linear(hidden_size, 4 * hidden_size)

    def forward(self, input, ht_minus_1, ct_minus_1):
        states_ih = self.linear_ih(input)
        states_hh = self.linear_hh(ht_minus_1)

        states = states_ih + states_hh

        i, f, g, o = torch.split(states, self.hidden_size, dim=-1)

        i_state = self.sigmoid(i)
        f_state = self.sigmoid(f)
        g_state = self.tanh(g)
        o_state = self.sigmoid(o)

        ct = ct_minus_1 * i_state + f_state * g_state
        ht = self.tanh(ct) * o_state

        return ht, ct


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.num_layers = num_layers

        self.lstm_blocks = nn.ModuleList(
            [LSTMblock(input_size, hidden_size)] +
            [LSTMblock(2 * hidden_size, hidden_size) for _ in range(num_layers - 1)] +
            [LSTMblock(input_size, hidden_size)] +
            [LSTMblock(2 * hidden_size, hidden_size) for _ in range(num_layers - 1)])

    def forward(self, input, h0=None, c0=None):

        if self.batch_first:
            input = input.transpose(1, 0)

        seq_len, batch_size = input.shape[0:2]
        num_layers = self.num_layers
        hidden_size = self.hidden_size
        device = next(self.parameters()).device

        ht_minus_1 = torch.zeros(batch_size, hidden_size, device=device)
        ct_minus_1 = torch.zeros(batch_size, hidden_size, device=device)

        for i in range(num_layers):
            ht_forward = []
            for j in range(seq_len):
                ht, ct = self.lstm_blocks[i](input[j], ht_minus_1, ct_minus_1)

                ht_forward.append(ht)
                ht_minus_1 = ht
                ct_minus_1 = ct

            ht_backward = []
            ht_minus_1.zero_()
            ct_minus_1.zero_()

            for j in range(-1, -seq_len - 1, -1):
                ht, ct = self.lstm_blocks[i + num_layers](input[j], ht_minus_1, ct_minus_1)

                ht_backward.append(ht)
                ht_minus_1 = ht
                ct_minus_1 = ct

            ht_minus_1.zero_()
            ct_minus_1.zero_()

            ht_forward = torch.stack(ht_forward)
            ht_backward = torch.flip(torch.stack(ht_backward), dims=(0,))

            input = torch.cat((ht_forward, ht_backward), dim=-1)

        outputs = input

        if self.batch_first:
            outputs = outputs.transpose(1, 0)

        return outputs

class LSTMNer(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super().__init__()

    self.lstm = BiLSTM(input_size, hidden_size, 2, True)
    self.linear = nn.Linear(2*hidden_size, num_classes)

  def forward(self, input):

    res = self.lstm(input)
    res = self.linear(res)

    return res


class CRF(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.num_labels = num_labels + 2  # num labels + (START, END)
        self.transition_scores = nn.Parameter(torch.empty(self.num_labels, self.num_labels))
        nn.init.xavier_uniform_(self.transition_scores)
        self.se_score = -1000

    def __call__(self, x, y):
        # x - BiLSTM outputs float[batch_size, seq_len, num_labels]
        # y - targets int[batch_size, seq_len]

        num_labels = self.num_labels
        device = y.device

        # Expand x and y with START and END
        x = self.expand_input(x)

        se_labels = torch.tensor([num_labels - 1]).repeat(y.shape[0], 1).to(device)
        y = torch.cat((se_labels - 1, y, se_labels), dim=-1)

        # Compute real path scores
        real_es = torch.gather(x, -1, y.unsqueeze(-1)).squeeze(-1).sum(-1)  # Emission score [batch_size]
        real_t = torch.stack((y[:, :-1], y[:, 1:]), dim=-1)
        real_ts = self.transition_scores[real_t[:, :, 0], real_t[:, :, 1]].sum(-1)  # Transition score [batch_size]
        real_path_scores = real_es + real_ts

        # Compute all paths scores
        x = x.transpose(1, 0)  # [seq_len, batch_size, num_labels]
        previous = x[0]
        seq_len = x.shape[0]
        # loop over seq_len, last previous is all_paths_score
        for i in range(1, seq_len):
            prev = previous.unsqueeze(-1)
            obs = x[i].unsqueeze(-2)
            scores = prev + obs + self.transition_scores  # [batch_size, num_labels, num_labels]
            scores = torch.logsumexp(scores, dim=-2)  # [batch_size, num_labels]
            previous = scores

        loss = -real_path_scores + torch.logsumexp(previous, dim=-1)

        return loss.sum() / loss.shape[0]

    def expand_input(self, x):
        num_labels = self.num_labels
        se_score = self.se_score
        device = x.device

        se_scores = torch.tensor([se_score]).repeat(*x.shape[0:2], 1).to(device)
        x = torch.cat((x, se_scores, se_scores), dim=-1)

        start_scores = torch.full((x.shape[0], 1, x.shape[2]), se_score).to(device)
        start_scores[:, :, num_labels - 2] = 0
        end_scores = torch.full((x.shape[0], 1, x.shape[2]), se_score).to(device)
        end_scores[:, :, num_labels - 1] = 0

        x = torch.cat((start_scores, x, end_scores), dim=-2)

        return x

    def predict(self, x):
        x = self.expand_input(x)
        x = x.transpose(1, 0)  # [seq_len, batch_size, num_labels]
        previous = x[0]

        alpha = []

        seq_len = x.shape[0]  # [batch_size, num_labels]
        for i in range(1, seq_len):
            prev = previous.unsqueeze(-1)
            obs = x[i].unsqueeze(-2)
            scores = prev + obs + self.transition_scores  # [batch_size, num_labels, num_labels]

            max_values = torch.max(scores, dim=-2)
            previous = max_values.values
            alpha.append(max_values.indices)

        alpha = torch.stack(alpha)  # [seq_len - 1, batch_size, num_labels]
        prev_label = previous.max(dim=-1).indices.unsqueeze(-1)  # [batch_size, 1]
        labels = prev_label

        for i in alpha.flip(0):
            prev_label = torch.gather(i, -1, prev_label)
            labels = torch.cat((prev_label, labels), -1)

        return labels[:, 1:-1]

class NerBiLSTMCRF(nn.Module):
  def __init__(self):
    super().__init__()
    num_classes = 18
    hidden_size = 768
    input_size = 300
    self.base_model = LSTMNer(input_size, hidden_size, num_classes)
    self.crf_layer = CRF(num_classes)

  def forward(self, input, target):
    emission_scores = self.base_model(input)
    loss = self.crf_layer(emission_scores, target)
    pred = self.crf_layer.predict(emission_scores)

    return loss, pred