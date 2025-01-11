import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):

        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        torch.cuda.empty_cache()
        return self.gamma * x + self.beta

class CrossAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(CrossAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        torch.cuda.empty_cache()
        return x.permute(0, 2, 1, 3)

    def forward(self, drugA, drugB):
        mixed_query_layer = self.query(drugA)
        mixed_key_layer = self.key(drugB)
        mixed_value_layer = self.value(drugB)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs_0 = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs_0)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        torch.cuda.empty_cache()
        return context_layer, attention_probs_0


class SelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        torch.cuda.empty_cache()
        return hidden_states

class Attention_CA(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Attention_CA, self).__init__()
        self.self = CrossAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = SelfOutput(hidden_size,hidden_dropout_prob)

    def forward(self, cell, drug):
        cell_self_output, cell_attention_probs_0 = self.self(cell, drug)
        drug_self_output, drug_attention_probs_0 = self.self(drug, cell)
        cell_attention_output = self.output(cell_self_output, cell)
        drug_attention_output = self.output(drug_self_output, drug)
        torch.cuda.empty_cache()
        return cell_attention_output, drug_attention_output, cell_attention_probs_0, drug_attention_probs_0

class Intermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(Intermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.relu(hidden_states)
        torch.cuda.empty_cache()
        return hidden_states


class Output(nn.Module):
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob):
        super(Output, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        torch.cuda.empty_cache()
        return hidden_states

class EncoderD2C(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                 hidden_dropout_prob):
        super(EncoderD2C, self).__init__()
        self.LayerNorm = LayerNorm(hidden_size)
        self.attention_CA = Attention_CA(hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob)
        self.intermediate = Intermediate(hidden_size, intermediate_size)
        self.output = Output(intermediate_size, hidden_size, hidden_dropout_prob)
        self.dense = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(intermediate_size, hidden_size))

    def forward(self, cell, drug):
        cell_1 = self.LayerNorm(cell)
        cell_attention_output, drug_attention_output, cell_attention_probs_0, drug_attention_probs_0 = self.attention_CA(cell_1, drug)
        cell_2 = cell_1 + cell_attention_output
        cell_3 = self.LayerNorm(cell_2)
        cell_4 = self.dense(cell_3)
        cell_layer_output = cell_2 + cell_4
        drug_intermediate_output = self.intermediate(drug_attention_output)
        drug_layer_output = self.output(drug_intermediate_output, drug_attention_output)
        torch.cuda.empty_cache()
        return cell_layer_output, drug_layer_output, cell_attention_probs_0, drug_attention_probs_0

class FeatureExpander(nn.Module):
    def __init__(self):
        super(FeatureExpander, self).__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=1)###确保Q、K用到
    def forward(self, x):
        x = self.conv(x)
        torch.cuda.empty_cache()
        return x

class DNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(DNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Linear(hidden_size // 2, 1)
        )

        self.drug_cell_CA = EncoderD2C(32, 32*2, 4, 0.1, 0.1)
        self.FeatureExpander = FeatureExpander()

        self.cell_fc = nn.Sequential(
            nn.Linear(16 * 32, 768),
            nn.ReLU()
        )
        self.drug_fc = nn.Sequential(
            nn.Linear(16 * 32,381),
            nn.ReLU()
        )
        self.cell = nn.Sequential(
            nn.Linear(768,16 * 32),
            nn.ReLU()
        )
        self.drug= nn.Sequential(
            nn.Linear(381, 16 * 32),
            nn.ReLU()
        )
    def forward(self, drug1_feat: torch.Tensor, drug2_feat: torch.Tensor, cell_feat: torch.Tensor):
        drug1 = self.drug(drug1_feat)
        drugA_feat = drug1.view(-1, 16, 32)
        drug2 = self.drug(drug2_feat)
        drugB_feat = drug2.view(-1, 16, 32)
        cell_feat = self.cell(cell_feat)
        cell = cell_feat.view(-1, 16, 32)
        cellA, drugA, cellA_attention, drugA_attention = self.drug_cell_CA(cell, drugA_feat)
        cellB, drugB, cellB_attention, drugB_attention = self.drug_cell_CA(cell, drugB_feat)
        cellA_feat = self.cell_fc(cellA.reshape(-1, cellA.shape[1] * cellA.shape[2]))
        drugA_feat = self.drug_fc(drugA.reshape(-1, drugA.shape[1] * drugA.shape[2]))
        cellB_feat = self.cell_fc(cellB.reshape(-1, cellB.shape[1] * cellB.shape[2]))
        drugB_feat = self.drug_fc(drugB.reshape(-1, drugB.shape[1] * drugB.shape[2]))
        feat = torch.cat([drugA_feat,drugB_feat, cellA_feat,cellB_feat], 1)
        out = self.network(feat)
        torch.cuda.empty_cache()
        return out
