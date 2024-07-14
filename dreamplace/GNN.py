import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
from dgl.nn.pytorch import GraphConv


# implement a two-layer GCN model using DGL
class GCN(torch.nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes, n_layers, activation):
        super(GCN, self).__init__()
        for i in range(n_layers-1):
            if i == 0:
                self.layers = torch.nn.ModuleList([GraphConv(in_feats, hidden_size, activation=activation, allow_zero_in_degree=True)])
            else:
                self.layers.append(GraphConv(hidden_size, hidden_size, activation=activation, allow_zero_in_degree=True))
        self.layers.append(GraphConv(hidden_size, num_classes))

    def forward(self, mfgs, inputs, train_state=False):
        res = inputs
        for i, layer in enumerate(self.layers):                
            h_dst = res[:mfgs[i].number_of_dst_nodes()]
            res = layer(mfgs[i], (res, h_dst))
          
        return res

class GAT(torch.nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes, n_layers, activation):
        super(GAT, self).__init__()
        for i in range(n_layers-1):
            if i == 0:
                self.layers = torch.nn.ModuleList([dgl.nn.GATConv(in_feats, hidden_size, num_heads=8, activation=activation)])
            else:
                self.layers.append(dgl.nn.GATConv(hidden_size * 8, hidden_size, num_heads=8, activation=activation))
        self.layers.append(dgl.nn.GATConv(hidden_size * 8, num_classes, num_heads=1))

    def forward(self, mfgs, inputs, remove_list=None):
        res = inputs
        for i, layer in enumerate(self.layers):                
            h_dst = res[:mfgs[i].number_of_dst_nodes()]
            res = layer(mfgs[i], (res, h_dst)).flatten(1)
        
        return res
    
class SAGE(torch.nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes, n_layers, activation):
        super(SAGE, self).__init__()
        for i in range(n_layers-1):
            if i == 0:
                self.layers = torch.nn.ModuleList([dgl.nn.SAGEConv(in_feats, hidden_size, 'gcn', activation=activation)])
            else:
                self.layers.append(dgl.nn.SAGEConv(hidden_size, hidden_size, 'gcn', activation=activation))
        self.layers.append(dgl.nn.SAGEConv(hidden_size, num_classes, 'gcn', activation=None))

    def forward(self, mfgs, inputs, remove_list=None):
        res = inputs
        for i, layer in enumerate(self.layers):                
            h_dst = res[:mfgs[i].number_of_dst_nodes()]
            res = layer(mfgs[i], (res, h_dst))
        
        return res

# implement a edgeGCN model using DGL
class EdgeUpdate(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(EdgeUpdate, self).__init__()
        self.fc = nn.Linear(in_feats * 2, out_feats)  # Fully connected layer for edge updates

    def forward(self, edges):
        # Concatenation of the two nodes that the edge connects
        h = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
        # Apply fully connected layer
        return {'h': self.fc(h)}

class NodeUpdate(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(NodeUpdate, self).__init__()
        self.fc = nn.Linear(in_feats, out_feats)  # Fully connected layer for node updates

    def forward(self, nodes):
        # Mean of all in- and outgoing edges
        h = torch.mean(nodes.mailbox['h'], dim=1)
        # Apply fully connected layer
        return {'h': self.fc(h)}

class EdgeGNN(nn.Module):
    def __init__(self, node_in_feats, edge_in_feats, hidden_size):
        super(EdgeGNN, self).__init__()
        self.edge_update = EdgeUpdate(node_in_feats, edge_in_feats)
        self.node_update = NodeUpdate(edge_in_feats, hidden_size)

    def forward(self, g, node_inputs):
        # Set initial node features
        g.ndata['h'] = node_inputs

        # Edge update
        g.apply_edges(self.edge_update)

        # Node update
        g.update_all(message_func=fn.copy_e('h', 'h'), reduce_func=self.node_update)
        return g.ndata['h']