import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads, dropout=0.2):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.out = nn.Sequential(nn.Linear(heads * self.head_dim, embed_size),
                                 nn.LayerNorm(embed_size),
                                 nn.Dropout(dropout))

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        out = self.out(out)

        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads, dropout)

        self.norm = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        query = self.norm(query)
        key = self.norm(key)
        value = self.norm(value)
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm(forward + x))

        return out
        
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
        
        self.drug_cell_CA = TransformerBlock(32, 4, 0.1, 4)

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
        cellA = self.drug_cell_CA(value=drugA_feat, key=drugA_feat, query=cell, mask=None)
        cellB = self.drug_cell_CA(value=drugB_feat, key=drugB_feat, query=cell, mask=None)
        drugA = self.drug_cell_CA(value=cell, key=cell, query=drugA_feat, mask=None)
        drugB = self.drug_cell_CA(value=cell, key=cell, query=drugB_feat, mask=None)
        cellA_feat = self.cell_fc(cellA.reshape(-1, cellA.shape[1] * cellA.shape[2]))
        drugA_feat = self.drug_fc(drugA.reshape(-1, drugA.shape[1] * drugA.shape[2]))
        cellB_feat = self.cell_fc(cellB.reshape(-1, cellB.shape[1] * cellB.shape[2]))
        drugB_feat = self.drug_fc(drugB.reshape(-1, drugB.shape[1] * drugB.shape[2]))
        feat = torch.cat([drugA_feat,drugB_feat, cellA_feat,cellB_feat], 1)
        out = self.network(feat)

        torch.cuda.empty_cache()
        return out
