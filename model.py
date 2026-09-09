import torch
import torch.nn as nn


def _make_tower(input_dim, hidden_dims, dropout):
    """A Linear/BatchNorm/ReLU/Dropout stack, without a final output layer -
    shared by MLP (which appends its own output layer) and TwoHeadMLP's
    per-branch towers (which get merged before the output layer instead)."""
    layers = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        prev_dim = hidden_dim
    return layers, prev_dim


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TwoHeadMLP(nn.Module):
    """Two independent input towers (e.g. mutation status, subtype flags),
    each with their own hidden layers, concatenated into a shared trunk that
    produces the final output. Giving each input block dedicated capacity
    before merging stops a large, noisy block (e.g. thousands of mutation
    features) from drowning out a small, clean one (e.g. subtype flags) in
    shared early hidden units - the failure mode observed when concatenating
    both blocks directly into a single MLP's input.

    forward() takes one already-concatenated input tensor (same calling
    convention as MLP, so it's a drop-in replacement in the training loop)
    and splits it internally at input_dim_a - callers just need to make sure
    block A's columns come first, which is how load_dataset() builds X."""

    def __init__(self, input_dim_a, input_dim_b, hidden_dims_a, hidden_dims_b, hidden_dims_shared, num_classes, dropout=0.3):
        super().__init__()
        self.input_dim_a = input_dim_a
        self.input_dim_b = input_dim_b

        layers_a, out_dim_a = _make_tower(input_dim_a, hidden_dims_a, dropout)
        layers_b, out_dim_b = _make_tower(input_dim_b, hidden_dims_b, dropout)
        self.branch_a = nn.Sequential(*layers_a) if layers_a else nn.Identity()
        self.branch_b = nn.Sequential(*layers_b) if layers_b else nn.Identity()
        self.out_dim_a = out_dim_a
        self.out_dim_b = out_dim_b

        shared_layers, prev_dim = _make_tower(out_dim_a + out_dim_b, hidden_dims_shared, dropout)
        shared_layers.append(nn.Linear(prev_dim, num_classes))
        self.shared = nn.Sequential(*shared_layers)

    def forward(self, x):
        x_a, x_b = x[:, :self.input_dim_a], x[:, self.input_dim_a:]
        h = torch.cat([self.branch_a(x_a), self.branch_b(x_b)], dim=1)
        return self.shared(h)

    def first_layer_a(self):
        """The mutation branch's first Linear layer, or None if branch_a has
        no hidden layers - used for the group-lasso input penalty and for
        transplanting pretrained subtype-only weights."""
        for m in self.branch_a.modules():
            if isinstance(m, nn.Linear):
                return m
        return None