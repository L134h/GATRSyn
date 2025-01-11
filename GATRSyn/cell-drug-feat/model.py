import torch
import torch.nn as nn
from torch.nn import LayerNorm
from torch_geometric.nn import BatchNorm
from Modified_GAT import GATConv as GATConv
from torch_geometric.nn import GraphSizeNorm
from model_utils import decide_loss_type


class GATEncoder(torch.nn.Module):

    def __init__(self, input_dim:int, output_dim:int, head_num, dropedge_rate, graph_dropout_rate, loss_type, with_edge, simple_distance, norm_type):

        super(GATEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_num = head_num
        self.dropedge_rate = dropedge_rate
        self.loss_type = loss_type
        self.simple_distance = simple_distance
        self.norm_type = norm_type
        self.conv = GATConv([input_dim, input_dim], output_dim, heads=head_num, dropout=dropedge_rate, with_edge=with_edge, simple_distance=simple_distance)

        if norm_type == "layer":
            self.bn = LayerNorm(output_dim*head_num)
            self.gbn = None
        else:
            self.bn = BatchNorm(output_dim*head_num)
            self.gbn = GraphSizeNorm()
        self.prelu = decide_loss_type(loss_type, output_dim*head_num)
        self.dropout_rate = graph_dropout_rate
        self.with_edge = with_edge

    def reset_parameters(self):
        self.conv.reset_parameters()
        self.bn.reset_parameters()

    def forward(self, edge_index ,x: torch.Tensor,edge_attr):

        if self.training:
            drop_node_mask = x.new_full((x.size(1),), 1 - self.dropout_rate, dtype=torch.float)
            drop_node_mask = torch.bernoulli(drop_node_mask)
            drop_node_mask = torch.reshape(drop_node_mask, (1, drop_node_mask.shape[0]))
            drop_node_feature = x * drop_node_mask

            drop_edge_mask = edge_attr.new_full((edge_attr.size(1),), 1 - self.dropout_rate, dtype=torch.float)
            drop_edge_mask = torch.bernoulli(drop_edge_mask)
            drop_edge_mask = torch.reshape(drop_edge_mask, (1, drop_edge_mask.shape[0]))
            drop_edge_attr = edge_attr * drop_edge_mask
        else:
            drop_node_feature = x
            drop_edge_attr = edge_attr

        torch.cuda.empty_cache()

        if self.with_edge == "Y":
            x_before, attention_value = self.conv((drop_node_feature, drop_node_feature), edge_index,
                                   edge_attr=drop_edge_attr, return_attention_weights=True)
            _, _, attention_value= attention_value.coo()
            attention = torch.reshape(attention_value, (1, attention_value.shape[0], attention_value.shape[1]))
        else:
            x_before, attention_value = self.conv((drop_node_feature, drop_node_feature), edge_index,
                                   edge_attr=None, return_attention_weights=True)
            _, _, attention_value = attention_value.coo()
            attention = torch.reshape(attention_value, (1, attention_value.shape[0], attention_value.shape[1]))
        if self.norm_type == "layer":
           temp = self.bn(x_before)
           out_x_temp = temp
        else:
            temp = self.gbn(self.bn(x_before))
            out_x_temp = temp

        x_after = self.prelu(out_x_temp)
        return x_after, attention



class Cell2Vec(torch.nn.Module):

    def __init__(self, encoder: GATEncoder, n_cell, n_dim):
        super(Cell2Vec, self).__init__()
        self.encoder = encoder
        self.embeddings = nn.Embedding(n_cell, n_dim)
        self.projector = nn.Sequential(
            nn.Linear(encoder.output_dim*2, n_dim),
            nn.Dropout()
        )
        self.edge_position_embedding = nn.Embedding(11, 32)
        self.edge_angle_embedding = nn.Embedding(11, 32)
        self.cell_cell_weight = nn.Parameter(torch.rand(n_dim,n_dim))

    def forward(self, edge_indices,x: torch.Tensor, edge_angle,
                x_indices: torch.LongTensor, c_indices: torch.LongTensor):
        drop_edge_attr_distance = edge_angle[:, 0]
        drop_edge_attr_distance = torch.div(drop_edge_attr_distance, 0.1, rounding_mode='trunc')
        drop_edge_attr_distance = drop_edge_attr_distance.type(torch.LongTensor)
        drop_edge_attr_distance = drop_edge_attr_distance.to(edge_angle.device)
        drop_edge_attr_angle = edge_angle[:, 1]
        drop_edge_attr_angle = torch.div(drop_edge_attr_angle, 0.1, rounding_mode='trunc')
        drop_edge_attr_angle = drop_edge_attr_angle.type(torch.LongTensor)
        drop_edge_attr_angle = drop_edge_attr_angle.to(edge_angle.device)

        drop_edge_attr_distance = self.edge_position_embedding(drop_edge_attr_distance)
        drop_edge_attr_angle = self.edge_angle_embedding(drop_edge_attr_angle)
        edge_angle = torch.cat((drop_edge_attr_distance, drop_edge_attr_angle), 1)

        encoded, attention = self.encoder(edge_indices, x, edge_angle)
        encoded = encoded.index_select(0, x_indices)
        proj = self.projector(encoded).permute(1, 0)
        emb = self.embeddings(c_indices)
        cell_cell = torch.sigmoid(torch.mm(torch.mm(emb, self.cell_cell_weight), self.embeddings.weight.t()))
        out = torch.mm(emb, proj)
        return out, attention, cell_cell
