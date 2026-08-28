import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, LayerNorm as GraphLayerNorm
from torch_geometric.nn import MessagePassing


# ===========================================================================
# NON-GNN BASELINES
# ===========================================================================

class SimpleMLP(nn.Module):
    """
    Feed-forward baseline: classifies edges using only their flow attributes
    (edge_attr). Does not use graph structure.
    It follows the shared graph-model interface but ignores graph topology,
    node count, global node IDs, and timestamps.
    """

    def __init__(self, edge_dim, hidden_dim, dropout=0.2, output_bias_init=None, node_dim=None):
        super().__init__()
        self.input_dim = edge_dim

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, 1)
        )

        if output_bias_init is not None:
            self.net[-1].bias.data.fill_(output_bias_init)

    def forward(self, edge_index, edge_attr, num_nodes,
                global_node_ids=None, timestamp=None):
        return self.net(edge_attr)


class EdgeGRU_Baseline(nn.Module):
    """
    Recurrent baseline: processes edge features with a GRU cell over time,
    maintaining per-node hidden states across graph windows.
    """

    def __init__(self, node_dim, edge_dim, hidden_dim, dropout, output_bias_init=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_memory = {}

        input_dim = (2 * node_dim) + edge_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        classifier_input_dim = (2 * hidden_dim) + edge_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        if output_bias_init is not None:
            self.classifier[-1].bias.data.fill_(output_bias_init)

    def manual_scatter_mean(self, src, index, dim_size):
        out = torch.zeros((dim_size, src.size(1)), device=src.device)
        out.index_add_(0, index, src)
        ones = torch.ones(src.size(0), 1, device=src.device)
        count = torch.zeros(dim_size, 1, device=src.device)
        count.index_add_(0, index, ones)
        count[count < 1] = 1
        return out / count

    def detach_all_memory(self):
        """Cut gradient flow through memory (Truncated BPTT)."""
        for k, v in self.node_memory.items():
            self.node_memory[k] = v.detach()

    def reset_memory(self):
        self.node_memory = {}

    def forward(self, x, edge_index, edge_attr, global_node_ids):
        device = x.device
        num_nodes_batch = x.size(0)

        src, dst = edge_index

        raw_features = torch.cat([x[src], x[dst], edge_attr], dim=1)
        encoded_features = self.encoder(raw_features)

        global_ids_list = global_node_ids.tolist()
        h_prev = torch.zeros(num_nodes_batch, self.hidden_dim, device=device)
        for i, gid in enumerate(global_ids_list):
            if gid in self.node_memory:
                h_prev[i] = self.node_memory[gid]

        aggr_input = self.manual_scatter_mean(encoded_features, src, dim_size=num_nodes_batch)
        h_new = self.gru(aggr_input, h_prev)

        h_new_stored = h_new.clone()
        for i, gid in enumerate(global_ids_list):
            self.node_memory[gid] = h_new_stored[i]

        edge_representation = torch.cat([h_new[src], h_new[dst], edge_attr], dim=1)
        return self.classifier(edge_representation)


# ===========================================================================
# STATIC GNN MODELS
# ===========================================================================

# ---------------------------------------------------------------------------
# EXPERIMENTAL VARIANTS (not used in final evaluation — kept for reference)
# ---------------------------------------------------------------------------

class StaticGNN(nn.Module):
    """
    [EXPERIMENTAL — superseded by StaticGNN_Identity]
    Basic GATv2 spatial GNN. Operates on raw node features without
    an identity projection step.
    """

    def __init__(self, node_dim, edge_dim, hidden_dim, out_dim, dropout):
        super(StaticGNN, self).__init__()

        self.gnn1 = GATv2Conv(
            in_channels=node_dim,
            out_channels=hidden_dim,
            edge_dim=edge_dim,
            heads=2,
            concat=False
        )
        self.gnn2 = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            edge_dim=edge_dim,
            heads=1,
            concat=False
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        if x.size(0) == 0: return torch.empty((0, 1), device=x.device)

        h1 = F.elu(self.gnn1(x, edge_index, edge_attr=edge_attr))
        h2 = F.elu(self.gnn2(h1, edge_index, edge_attr=edge_attr))

        src, dst = edge_index
        edge_rep = torch.cat([h2[src], h2[dst], edge_attr], dim=1)
        return self.classifier(edge_rep)


class StaticGNN_biasinit(nn.Module):
    """
    [EXPERIMENTAL — superseded by StaticGNN_Identity]
    StaticGNN with custom output bias initialization and fixed out_dim=1.
    """

    def __init__(self, node_dim, edge_dim, hidden_dim, dropout, output_bias_init=None):
        super(StaticGNN_biasinit, self).__init__()

        self.gnn1 = GATv2Conv(
            in_channels=node_dim,
            out_channels=hidden_dim,
            edge_dim=edge_dim,
            heads=2,
            concat=False
        )
        self.gnn2 = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            edge_dim=edge_dim,
            heads=1,
            concat=False
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        if output_bias_init is not None:
            self.classifier[-1].bias.data.fill_(output_bias_init)

    def forward(self, x, edge_index, edge_attr):
        if x.size(0) == 0: return torch.empty((0, 1), device=x.device)

        h1 = F.elu(self.gnn1(x, edge_index, edge_attr=edge_attr))
        h2 = F.elu(self.gnn2(h1, edge_index, edge_attr=edge_attr))

        src, dst = edge_index
        edge_rep = torch.cat([h2[src], h2[dst], edge_attr], dim=1)
        return self.classifier(edge_rep)


# ===========================================================================
# EXTERNAL BASELINE
# ===========================================================================

class EdgeGRU_Baseline_NoX(nn.Module):
    """
    Temporal baseline without node features.

    Encodes only edge_attr (no dummy node vectors), then aggregates per
    source node and updates a per-node GRU hidden state across windows.
    This is the clean "time-only" ablation: temporal memory without any
    graph-convolution or node-feature signal.

    No persisted node-feature tensor is required. ``num_nodes`` defines the
    graph-local state shape and ``global_node_ids`` provides stable temporal
    memory keys.
    """

    def __init__(self, edge_dim, hidden_dim, dropout, output_bias_init=None, node_dim=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_memory = {}

        self.encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        if output_bias_init is not None:
            self.classifier[-1].bias.data.fill_(output_bias_init)

    def manual_scatter_mean(self, src, index, dim_size):
        out = torch.zeros((dim_size, src.size(1)), device=src.device)
        out.index_add_(0, index, src)
        ones = torch.ones(src.size(0), 1, device=src.device)
        count = torch.zeros(dim_size, 1, device=src.device)
        count.index_add_(0, index, ones)
        count[count < 1] = 1
        return out / count

    def detach_all_memory(self):
        for k, v in self.node_memory.items():
            self.node_memory[k] = v.detach()

    def reset_memory(self):
        self.node_memory = {}

    def forward(self, edge_index, edge_attr, num_nodes,
                global_node_ids=None, timestamp=None):
        if global_node_ids is None:
            raise ValueError("EdgeGRU_Baseline_NoX requires global_node_ids.")
        device = edge_attr.device
        num_nodes_batch = int(num_nodes)
        if global_node_ids.numel() != num_nodes_batch:
            raise ValueError("global_node_ids must contain one ID per local node.")

        src, dst = edge_index

        encoded_features = self.encoder(edge_attr)

        global_ids_list = global_node_ids.tolist()
        h_prev = torch.zeros(num_nodes_batch, self.hidden_dim, device=device)
        for i, gid in enumerate(global_ids_list):
            if gid in self.node_memory:
                h_prev[i] = self.node_memory[gid]

        aggr_input = self.manual_scatter_mean(encoded_features, src, dim_size=num_nodes_batch)
        h_new = self.gru(aggr_input, h_prev)

        h_new_stored = h_new.clone()
        for i, gid in enumerate(global_ids_list):
            self.node_memory[gid] = h_new_stored[i]

        edge_representation = torch.cat([h_new[src], h_new[dst], edge_attr], dim=1)
        return self.classifier(edge_representation)


# ===========================================================================
# EXTERNAL BASELINE
# ===========================================================================

class _ESAGEConv(MessagePassing):
    """
    Single E-GraphSAGE layer (Lo et al. 2022, Algorithm 1).

    Message:     each edge sends its own features e_uv (no neighbor node embedding).
    Aggregation: mean of incident edge features at node v.
    Update:      Linear(concat(h_v, mean_edges)) → ReLU.

    Linear input is in_channels + edge_dim because the aggregated vector
    has size edge_dim (edge features only, not neighbor embeddings).
    """

    def __init__(self, in_channels, out_channels, edge_dim):
        super().__init__(aggr='mean', flow='source_to_target')
        self.lin = nn.Linear(in_channels + edge_dim, out_channels)

    def forward(self, x, edge_index, edge_attr):
        n = x.size(0)
        aggr = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=(n, n))
        return F.relu(self.lin(torch.cat([x, aggr], dim=1)))

    def message(self, edge_attr):
        return edge_attr

    def update(self, aggr_out, x=None):
        return aggr_out


class E_GraphSAGE(nn.Module):
    """
    E-GraphSAGE adapted for edge (flow) classification.

    Message passing follows Lo et al. 2022 (Alg. 1): aggregates incident
    edge features without including neighbor node embeddings.
    Classifier: concat(h_src, h_dst, edge_attr) → MLP, following
    Chang & Branco 2021 (arXiv:2111.13597) to preserve edge features
    that are diluted during aggregation. hidden_dim=32 for equal-capacity
    comparison.

    The graph artifacts have no node features. This implementation constructs
    its documented all-ones constant initial node state internally.
    """

    def __init__(self, node_dim, edge_dim, hidden_dim, dropout=0.2, output_bias_init=None):
        super().__init__()
        self.node_dim = node_dim
        self.dropout_rate = dropout

        self.sage1 = _ESAGEConv(node_dim,    hidden_dim, edge_dim)
        self.sage2 = _ESAGEConv(hidden_dim,  hidden_dim, edge_dim)

        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        if output_bias_init is not None:
            self.classifier[-1].bias.data.fill_(output_bias_init)

    def forward(self, edge_index, edge_attr, num_nodes,
                global_node_ids=None, timestamp=None):
        num_nodes = int(num_nodes)
        if num_nodes == 0:
            return torch.empty((0, 1), device=edge_attr.device)

        x = edge_attr.new_ones((num_nodes, self.node_dim))

        h = F.dropout(self.sage1(x, edge_index, edge_attr),
                      p=self.dropout_rate, training=self.training)
        h = self.sage2(h, edge_index, edge_attr)

        src, dst = edge_index
        edge_rep = torch.cat([h[src], h[dst], edge_attr], dim=1)
        return self.classifier(edge_rep)


# ---------------------------------------------------------------------------
# FINAL MODELS
# ---------------------------------------------------------------------------

class StaticGNN_Identity(nn.Module):
    """
    Static spatial GNN with identity projection.
    Builds node features from edge aggregation (what each IP sends / receives),
    then runs two GATv2 layers for edge classification.
    """

    def __init__(self, node_dim, edge_dim, hidden_dim, dropout=0.2, output_bias_init=None):
        super(StaticGNN_Identity, self).__init__()

        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout

        # IDENTITY PROJECTION
        self.edge_proj = nn.Linear(edge_dim, node_dim)

        gnn_input_dim = 2 * node_dim

        self.gnn1 = GATv2Conv(
            in_channels=gnn_input_dim, out_channels=hidden_dim,
            edge_dim=edge_dim, heads=2, concat=False, dropout=dropout
        )
        self.norm1 = GraphLayerNorm(hidden_dim)

        self.gnn2 = GATv2Conv(
            in_channels=hidden_dim, out_channels=hidden_dim,
            edge_dim=edge_dim, heads=1, concat=False, dropout=dropout
        )
        self.norm2 = GraphLayerNorm(hidden_dim)

        classifier_input_dim = (2 * hidden_dim) + edge_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        if output_bias_init is not None:
            self.classifier[-1].bias.data.fill_(output_bias_init)

    def manual_scatter_mean(self, src, index, dim_size):
        out = torch.zeros((dim_size, src.size(1)), device=src.device)
        out.index_add_(0, index, src)
        ones = torch.ones(src.size(0), 1, device=src.device)
        count = torch.zeros(dim_size, 1, device=src.device)
        count.index_add_(0, index, ones)
        count[count < 1] = 1
        return out / count

    def forward(self, edge_index, edge_attr, num_nodes,
                global_node_ids=None, timestamp=None):
        num_nodes = int(num_nodes)
        if num_nodes == 0:
            return torch.empty((0, 1), device=edge_attr.device)

        # STEP 1: Build node identity from edge aggregation
        edge_embeddings = F.relu(self.edge_proj(edge_attr))
        src, dst = edge_index
        x_out = self.manual_scatter_mean(edge_embeddings, src, dim_size=num_nodes)
        x_in  = self.manual_scatter_mean(edge_embeddings, dst, dim_size=num_nodes)
        x_input = torch.cat([x_out, x_in], dim=1)

        # STEP 2: GNN layers
        h1 = F.dropout(F.elu(self.norm1(self.gnn1(x_input, edge_index, edge_attr=edge_attr))),
                       p=self.dropout_rate, training=self.training)
        h2 = F.elu(self.norm2(self.gnn2(h1, edge_index, edge_attr=edge_attr)))

        # STEP 3: Edge classification
        src, dst = edge_index
        edge_rep = torch.cat([h2[src], h2[dst], edge_attr], dim=1)
        return self.classifier(edge_rep)


# ===========================================================================
# SPATIAL-TEMPORAL GNN MODELS (ST-GNN)
# ===========================================================================

# ---------------------------------------------------------------------------
# EXPERIMENTAL VARIANTS (not used in final evaluation — kept for reference)
# ---------------------------------------------------------------------------

class ST_GNN(nn.Module):
    """
    [EXPERIMENTAL — superseded by ST_GNN_Identity]
    Basic spatial-temporal GNN: GATv2 spatial layers + GRU temporal memory,
    operating on raw node features without identity projection.
    """

    def __init__(self, node_dim, edge_dim, hidden_dim, out_dim, dropout):
        super(ST_GNN, self).__init__()
        self.hidden_dim = hidden_dim

        self.gnn1 = GATv2Conv(
            in_channels=node_dim, out_channels=hidden_dim,
            edge_dim=edge_dim, heads=2, concat=False
        )
        self.gnn2 = GATv2Conv(
            in_channels=hidden_dim, out_channels=hidden_dim,
            edge_dim=edge_dim, heads=1, concat=False
        )
        self.gru = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_dim)
        )
        self.node_memory = {}

    def get_memory(self, ids, device):
        return torch.stack([
            self.node_memory.get(i.item(), torch.zeros(self.hidden_dim, device=device))
            for i in ids
        ])

    def update_memory(self, ids, h_new):
        h_detached = h_new.detach()
        for idx, gid in enumerate(ids):
            self.node_memory[gid.item()] = h_detached[idx]

    def detach_all_memory(self):
        for k in self.node_memory:
            self.node_memory[k] = self.node_memory[k].detach()

    def forward(self, x, edge_index, edge_attr, global_ids):
        if x.size(0) == 0: return torch.empty((0, 1), device=x.device)

        h_prev = self.get_memory(global_ids, x.device)

        z = F.elu(self.gnn1(x, edge_index, edge_attr=edge_attr))
        z = F.elu(self.gnn2(z, edge_index, edge_attr=edge_attr))

        h_current = self.gru(z, h_prev)
        self.update_memory(global_ids, h_current)

        src, dst = edge_index
        edge_rep = torch.cat([h_current[src], h_current[dst], edge_attr], dim=1)
        return self.classifier(edge_rep)


class ST_GNN_biasinit(nn.Module):
    """
    [EXPERIMENTAL — superseded by ST_GNN_Identity]
    ST_GNN with custom output bias initialization and fixed out_dim=1.
    """

    def __init__(self, node_dim, edge_dim, hidden_dim, dropout, output_bias_init=None):
        super(ST_GNN_biasinit, self).__init__()
        self.hidden_dim = hidden_dim

        self.gnn1 = GATv2Conv(
            in_channels=node_dim, out_channels=hidden_dim,
            edge_dim=edge_dim, heads=2, concat=False
        )
        self.gnn2 = GATv2Conv(
            in_channels=hidden_dim, out_channels=hidden_dim,
            edge_dim=edge_dim, heads=1, concat=False
        )
        self.gru = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        if output_bias_init is not None:
            self.classifier[-1].bias.data.fill_(output_bias_init)

        self.node_memory = {}

    def get_memory(self, ids, device):
        return torch.stack([
            self.node_memory.get(i.item(), torch.zeros(self.hidden_dim, device=device))
            for i in ids
        ])

    def update_memory(self, ids, h_new):
        h_detached = h_new.detach()
        for idx, gid in enumerate(ids):
            self.node_memory[gid.item()] = h_detached[idx]

    def detach_all_memory(self):
        for k in self.node_memory:
            self.node_memory[k] = self.node_memory[k].detach()

    def forward(self, x, edge_index, edge_attr, global_ids):
        if x.size(0) == 0: return torch.empty((0, 1), device=x.device)

        h_prev = self.get_memory(global_ids, x.device)

        z = F.elu(self.gnn1(x, edge_index, edge_attr=edge_attr))
        z = F.elu(self.gnn2(z, edge_index, edge_attr=edge_attr))

        h_current = self.gru(z, h_prev)
        self.update_memory(global_ids, h_current)

        src, dst = edge_index
        edge_rep = torch.cat([h_current[src], h_current[dst], edge_attr], dim=1)
        return self.classifier(edge_rep)


class ST_GNN_Robust(nn.Module):
    """
    [EXPERIMENTAL — superseded by ST_GNN_Identity]
    ST_GNN_biasinit with GraphLayerNorm after each GNN layer for training stability.
    """

    def __init__(self, node_dim, edge_dim, hidden_dim, dropout=0.2, output_bias_init=None):
        super(ST_GNN_Robust, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout

        self.gnn1 = GATv2Conv(
            in_channels=node_dim, out_channels=hidden_dim,
            edge_dim=edge_dim, heads=2, concat=False, dropout=dropout
        )
        self.norm1 = GraphLayerNorm(hidden_dim)

        self.gnn2 = GATv2Conv(
            in_channels=hidden_dim, out_channels=hidden_dim,
            edge_dim=edge_dim, heads=1, concat=False, dropout=dropout
        )
        self.norm2 = GraphLayerNorm(hidden_dim)

        self.gru = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)

        classifier_input_dim = (2 * hidden_dim) + edge_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        if output_bias_init is not None:
            self.classifier[-1].bias.data.fill_(output_bias_init)

        self.node_memory = {}

    def get_memory(self, ids, device):
        return torch.stack([
            self.node_memory.get(i.item(), torch.zeros(self.hidden_dim, device=device))
            for i in ids
        ])

    def update_memory(self, ids, h_new):
        h_detached = h_new.detach()
        for idx, gid in enumerate(ids):
            self.node_memory[gid.item()] = h_detached[idx]

    def detach_all_memory(self):
        for k in self.node_memory:
            self.node_memory[k] = self.node_memory[k].detach()

    def reset_memory(self):
        self.node_memory = {}

    def forward(self, x, edge_index, edge_attr, global_ids):
        if x.size(0) == 0: return torch.empty((0, 1), device=x.device)

        h_prev = self.get_memory(global_ids, x.device)

        z = self.gnn1(x, edge_index, edge_attr=edge_attr)
        z = F.dropout(F.elu(self.norm1(z)), p=self.dropout_rate, training=self.training)

        z = self.gnn2(z, edge_index, edge_attr=edge_attr)
        z = F.elu(self.norm2(z))

        h_current = self.gru(z, h_prev)
        self.update_memory(global_ids, h_current)

        src, dst = edge_index
        edge_rep = torch.cat([h_current[src], h_current[dst], edge_attr], dim=1)
        return self.classifier(edge_rep)


# ---------------------------------------------------------------------------
# FINAL MODELS
# ---------------------------------------------------------------------------

class ST_GNN_Identity(nn.Module):
    """
    Spatial-temporal GNN with identity projection.
    Builds node features from edge aggregation (what each IP sends / receives),
    then runs two GATv2 layers followed by a GRU for temporal memory.
    """

    def __init__(self, node_dim, edge_dim, hidden_dim, dropout=0.2, output_bias_init=None):
        super(ST_GNN_Identity, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout

        # IDENTITY PROJECTION
        self.edge_proj = nn.Linear(edge_dim, node_dim)

        gnn_input_dim = 2 * node_dim

        self.gnn1 = GATv2Conv(
            in_channels=gnn_input_dim, out_channels=hidden_dim,
            edge_dim=edge_dim, heads=2, concat=False, dropout=dropout
        )
        self.norm1 = GraphLayerNorm(hidden_dim)

        self.gnn2 = GATv2Conv(
            in_channels=hidden_dim, out_channels=hidden_dim,
            edge_dim=edge_dim, heads=1, concat=False, dropout=dropout
        )
        self.norm2 = GraphLayerNorm(hidden_dim)

        self.gru = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)

        classifier_input_dim = (2 * hidden_dim) + edge_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        if output_bias_init is not None:
            self.classifier[-1].bias.data.fill_(output_bias_init)

        self.node_memory = {}

    def manual_scatter_mean(self, src, index, dim_size):
        out = torch.zeros((dim_size, src.size(1)), device=src.device)
        out.index_add_(0, index, src)
        ones = torch.ones(src.size(0), 1, device=src.device)
        count = torch.zeros(dim_size, 1, device=src.device)
        count.index_add_(0, index, ones)
        count[count < 1] = 1
        return out / count

    def get_memory(self, ids, device):
        return torch.stack([
            self.node_memory.get(i.item(), torch.zeros(self.hidden_dim, device=device))
            for i in ids
        ])

    def update_memory(self, ids, h_new):
        h_stored = h_new.clone()
        for idx, gid in enumerate(ids):
            self.node_memory[gid.item()] = h_stored[idx]

    def detach_all_memory(self):
        for k in self.node_memory:
            self.node_memory[k] = self.node_memory[k].detach()

    def reset_memory(self):
        self.node_memory = {}

    def forward(self, edge_index, edge_attr, num_nodes,
                global_node_ids=None, timestamp=None):
        num_nodes = int(num_nodes)
        if num_nodes == 0:
            return torch.empty((0, 1), device=edge_attr.device)
        if global_node_ids is None:
            raise ValueError("ST_GNN_Identity requires global_node_ids.")
        if global_node_ids.numel() != num_nodes:
            raise ValueError("global_node_ids must contain one ID per local node.")

        # STEP 1: Build node identity from edge aggregation
        edge_embeddings = F.relu(self.edge_proj(edge_attr))
        src, dst = edge_index
        x_out = self.manual_scatter_mean(edge_embeddings, src, dim_size=num_nodes)
        x_in  = self.manual_scatter_mean(edge_embeddings, dst, dim_size=num_nodes)
        x_input = torch.cat([x_out, x_in], dim=1)

        # STEP 2: GNN layers + GRU memory
        h_prev = self.get_memory(global_node_ids, edge_attr.device)

        z = self.gnn1(x_input, edge_index, edge_attr=edge_attr)
        z = F.dropout(F.elu(self.norm1(z)), p=self.dropout_rate, training=self.training)

        z = self.gnn2(z, edge_index, edge_attr=edge_attr)
        z = F.elu(self.norm2(z))

        h_current = self.gru(z, h_prev)
        self.update_memory(global_node_ids, h_current)

        # STEP 3: Edge classification
        edge_rep = torch.cat([h_current[src], h_current[dst], edge_attr], dim=1)
        return self.classifier(edge_rep)

