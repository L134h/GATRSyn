import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):

        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x): ###对X的最后一个维度进行归一化，应用偏置和缩放
        u = x.mean(-1, keepdim=True)##对第二个维数取均值
        # Normalize input_tensor
        s = (x - u).pow(2).mean(-1, keepdim=True)
        # Apply scaling and bias
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)##归一化 第二列维数768

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

    def transpose_for_scores(self, x):###为了便于多头计算
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)

        torch.cuda.empty_cache()
        return x.permute(0, 2, 1, 3)###为了多头运算，需要加入序列维度

    def forward(self, drugA, drugB):
        # ##update drugA
        # query = nn.Linear(drugA.size(-1), self.all_head_size).to('cuda:0')##768
        # key = nn.Linear(drugB.size(-1), self.all_head_size).to('cuda:0')
        # value = nn.Linear(drugB.size(-1), self.all_head_size).to('cuda:0')##762
        #
        # mixed_query_layer = query(drugA)
        # mixed_key_layer = key(drugB)
        # mixed_value_layer = value(drugB)
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

        context_layer = torch.matmul(attention_probs, value_layer)####论文中的计算公式
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
        # dense = nn.Linear(hidden_states.size(-1), hidden_states.size(-1)).to('cuda:0')
        # layerNorm = LayerNorm(hidden_states.size(-1)).to('cuda:0')

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
        cell_self_output, cell_attention_probs_0 = self.self(cell, drug) ##768，768
        drug_self_output, drug_attention_probs_0 = self.self(drug, cell)##(256,1,768)
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

# Drug-cell mutual-attention encoder
class EncoderD2C(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                 hidden_dropout_prob):
        super(EncoderD2C, self).__init__()
        self.LayerNorm = LayerNorm(hidden_size)
        self.attention_CA = Attention_CA(hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob)
        self.intermediate = Intermediate(hidden_size, intermediate_size)
        self.output = Output(intermediate_size, hidden_size, hidden_dropout_prob)#768,384,0.1
        self.dense = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(intermediate_size, hidden_size))

    def forward(self, cell, drug):
        cell_1 = self.LayerNorm(cell)
        cell_attention_output, drug_attention_output, cell_attention_probs_0, drug_attention_probs_0 = self.attention_CA(cell_1, drug)
        # cell_output
        cell_2 = cell_1 + cell_attention_output
        cell_3 = self.LayerNorm(cell_2)
        cell_4 = self.dense(cell_3)
        cell_layer_output = cell_2 + cell_4
        # drug_output
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
            nn.Linear(hidden_size, hidden_size // 2),  ##第二个线性层是第一个神经元数量一半
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Linear(hidden_size // 2, 2)##回归：最后一层神经元数量仅有一个；分类为两个
        )  ##C图中的网络，三个线性层，前两个跟着relu激活函数外加batchnorm（批量归一化）层

        self.drug_cell_CA = EncoderD2C(32, 32*2, 4, 0.1, 0.1)
        # self.drug_cell_CA = EncoderD2C(512, 512*2, 4, 0.1, 0.1)
        self.FeatureExpander = FeatureExpander()

        self.cell_fc = nn.Sequential(
            # nn.Linear(512 * 64, 768),
            nn.Linear(16 * 32, 768),
            nn.ReLU()
        )
        self.drug_fc = nn.Sequential(
            # nn.Linear(512 * 64, 381),
            nn.Linear(16 * 32,376),
            nn.ReLU()
        )
        self.cell = nn.Sequential(
            # nn.Linear(768, 64*512),
            nn.Linear(768,16 * 32),
            nn.ReLU()
        )
        self.drug= nn.Sequential(
            # nn.Linear(381, 64*512),
            nn.Linear(376, 16 * 32),
            nn.ReLU()
        )
    def forward(self, drug1_feat: torch.Tensor, drug2_feat: torch.Tensor, cell_feat: torch.Tensor):
        ##reshpe
        drug1 = self.drug(drug1_feat)
        # drug1_sequence = drug1.unsqueeze(1)  # 在第二个维度上添加序列维度
        # drugA_feat = self.FeatureExpander(drug1_sequence)
        drugA_feat = drug1.view(-1, 16, 32)
        drug2 = self.drug(drug2_feat)
        # drug2_sequence = drug2.unsqueeze(1)  # 在第二个维度上添加序列维度
        # drugB_feat = self.FeatureExpander(drug2_sequence)
        drugB_feat = drug2.view(-1, 16, 32)
        cell_feat = self.cell(cell_feat)
        # cell_feat_sequence = cell_feat.unsqueeze(1)  # 在第二个维度上添加序列维度
        # cell = self.FeatureExpander(cell_feat_sequence)
        cell = cell_feat.view(-1, 16, 32)
        cellA, drugA, cellA_attention, drugA_attention = self.drug_cell_CA(cell, drugA_feat)
        cellB, drugB, cellB_attention, drugB_attention = self.drug_cell_CA(cell, drugB_feat)
        cellA_feat = self.cell_fc(cellA.reshape(-1, cellA.shape[1] * cellA.shape[2]))
        drugA_feat = self.drug_fc(drugA.reshape(-1, drugA.shape[1] * drugA.shape[2]))
        cellB_feat = self.cell_fc(cellB.reshape(-1, cellB.shape[1] * cellB.shape[2]))
        drugB_feat = self.drug_fc(drugB.reshape(-1, drugB.shape[1] * drugB.shape[2]))
        feat = torch.cat([drugA_feat, drugB_feat,cellA_feat,cellB_feat], 1)##(253+128+253+128)+768  drugA_feat, drugB_feat,drugA_feat
        out = self.network(feat)

        # ####repeat
        # # drug1 = self.drug(drug1_feat)
        # # drug1_sequence = drug1.unsqueeze(1)  # 在第二个维度上添加序列维度
        # # drugA_feat = self.FeatureExpander(drug1_sequence)
        # # drug2 = self.drug(drug2_feat)
        # # drug2_sequence = drug2.unsqueeze(1)  # 在第二个维度上添加序列维度
        # # drugB_feat = self.FeatureExpander(drug2_sequence)
        # # cell_feat = self.cell(cell_feat)
        # # cell_feat_sequence = cell_feat.unsqueeze(1)  # 在第二个维度上添加序列维度
        # # cell = self.FeatureExpander(cell_feat_sequence)
        # # cellA, drugA, cellA_attention, drugA_attention = self.drug_cell_CA(cell, drugA_feat)
        # # cellB, drugB, cellB_attention, drugB_attention = self.drug_cell_CA(cell, drugB_feat)
        # # # cellA_feat = self.cell_fc(cellA.view(-1, cellA.shape[1] * cellA.shape[2]))
        # cellA_feat = self.cell_fc(cellA.reshape(-1, cellA.shape[1] * cellA.shape[2]))
        # drugA_feat = self.drug_fc(drugA.reshape(-1, drugA.shape[1] * drugA.shape[2]))
        # cellB_feat = self.cell_fc(cellB.reshape(-1, cellB.shape[1] * cellB.shape[2]))
        # drugB_feat = self.drug_fc(drugB.reshape(-1, drugB.shape[1] * drugB.shape[2]))
        # feat = torch.cat([drugA_feat,drugB_feat, cellA_feat,cellB_feat], 1)##(253+128+253+128)+768
        # out = self.network(feat)

        #  ####全连接层、线性变换
        # drug1 = self.drug(drug1_feat)
        # # drug1_feat_flatten = self.FeatureExpander(drug1)
        # drugA_feat = drug1.view(drug1.size()[0],-1,512)
        # drug2 = self.drug(drug2_feat)
        # # drug2_feat_flatten = self.FeatureExpander(drug2 )
        # drugB_feat = drug2.view(drug2.size()[0],-1, 512)
        # cell_feat = self.cell(cell_feat)
        # # cell_feat_flatten = self.FeatureExpander(cell_feat)
        # cell = cell_feat.view(cell_feat.size()[0],-1, 512)
        # cellA, drugA, cellA_attention, drugA_attention = self.drug_cell_CA(cell, drugA_feat)
        # cellB, drugB, cellB_attention, drugB_attention = self.drug_cell_CA(cell, drugB_feat)
        # # # cellA_feat = self.cell_fc(cellA.view(-1, cellA.shape[1] * cellA.shape[2]))
        # cellA_feat = self.cell_fc(cellA.reshape(-1, cellA.shape[1] * cellA.shape[2]))
        # drugA_feat = self.drug_fc(drugA.reshape(-1, drugA.shape[1] * drugA.shape[2]))
        # cellB_feat = self.cell_fc(cellB.reshape(-1, cellB.shape[1] * cellB.shape[2]))
        # drugB_feat = self.drug_fc(drugB.reshape(-1, drugB.shape[1] * drugB.shape[2]))
        # feat = torch.cat([drugA_feat,drugB_feat, cellA_feat,cellB_feat], 1)##(253+128+253+128)+768
        # out = self.network(feat)

        # ###原始
        # feat = torch.cat([drug1_feat, drug2_feat, cell_feat], 1)
        # out = self.network(feat)
        torch.cuda.empty_cache()
        return out
















    # def forward(self, drug1_feat: torch.Tensor, drug2_feat: torch.Tensor, cell_feat: torch.Tensor):
    #     # # drugcomb = torch.cat([drug1_feat, drug2_feat],1)##253+128
    #     # # drugcomb = drugcomb.to(cell_feat.device)  # 将drugcomb转移到和cell_feat相同的设备上
    #     # padding = cell_feat.size(-1) - drugcomb.size(-1) # 计算需要填充的数量     # 使用 torch.nn.functional.pad 进行填充
    #     # drugcomb_feat = torch.nn.functional.pad(drugcomb, (0, padding), 'constant', 0)##将其和cell特征变到同一维度
    #     #
    #     # drugcomb_sequence = drugcomb_feat.unsqueeze(1)  # 在第二个维度上添加序列维度
    #     # drugs = drugcomb_sequence.repeat(1, 1, 1)
    #     #
    #     # cell_feat_sequence = cell_feat.unsqueeze(1)  # 在第二个维度上添加序列维度
    #     # cell = cell_feat_sequence.repeat(1, 1, 1)
    #     # drugcomb_feat = np.pad(cell_feat, ((0, 0), (0, cell_feat.size(-1) - drugcomb.size(-1))), 'constant', constant_values=0)
    #     cell, drug, cell_attention, drug_attention = self.drug_cell_CA(cell, drugs) ##384+384=768 ,253+128=381*2
    #
    #     cell_feat = self.cell_fc(cell.view(-1, cell.shape[1] * cell.shape[2]))
    #     drugs_feat = self.drug_fc(drug.view(-1, drug.shape[1] * drug.shape[2]))##还原原始维度
    #     feat = torch.cat([drugs_feat, cell_feat], 1)##(253+128+253+128)+768
    #     out = self.network(feat)
    #     return out  ###拼接药物和细胞系特征输入网络进行预测
