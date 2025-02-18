import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from moe_module import *
from itertools import combinations

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation=nn.ReLU(), dropout=0.5):
        super(MLP, self).__init__()
        layers = []
        self.drop = nn.Dropout(dropout)
        if num_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation)
            layers.append(self.drop)
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(activation)
                layers.append(self.drop)
            layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class Custom3DCNN(nn.Module):
    #Architecture provided by: End-To-End Alzheimer's Disease Diagnosis and Biomarker Identification
    def __init__(self, hidden_dim=128):
        super(Custom3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv3d(32, 32, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)
        self.dropout1 = nn.Dropout3d(0.2)


        self.conv3 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv4 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=3)
        self.dropout2 = nn.Dropout3d(0.2)

        self.conv5 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv6 = nn.Conv3d(128, hidden_dim, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=(4, 4, 4))
        self.dropout3 = nn.Dropout3d(0.2)

        # Flatten the output and add a fully connected layer to reduce to hidden_dim
        self.fc = nn.Linear(hidden_dim * 3 * 3 * 4, hidden_dim)
        
        # self.fc1 = nn.Linear(128*6*6, 512)
        # self.fc2 = nn.Linear(512, 256)
        # self.fc3 = nn.Linear(256, num_classes)
        # self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout1(self.pool1(x))

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.dropout2(self.pool2(x))

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.dropout3(self.pool3(x))


        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc(x)  # Apply the fully connected layer
        
        # x = x.view(x.size(0), -1)
        # x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        # x = F.relu(self.fc2(x))
        # x = self.dropout(x)
        # x = self.fc3(x)

        return x

class FusionMLP(nn.Module):
    def __init__(self, total_input_dim, hidden_dim, output_dim, num_layers):
        super(FusionMLP, self).__init__()
        layers = []
        layers.append(nn.Linear(total_input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, *inputs):
        x = torch.cat(inputs, dim=1)
        return self.network(x)
    

class MoE_Retriever(nn.Module):
    def __init__(self, num_modalities, num_patches, hidden_dim, num_experts, num_routers, top_k, num_heads=2, dropout=0.5):
        super(MoE_Retriever, self).__init__()
        layers = []
        layers.append(TransformerEncoderLayer(num_experts, num_routers, hidden_dim, num_head=num_heads, dropout=dropout, hidden_times=2, mlp_sparse=True, top_k=top_k))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs, expert_indices=None, num_intra=1, num_inter=1, inter_weight=0.5):
        # expert_indices become constraint for MoE-Retriever
        x = self.network[-1](inputs, expert_indices)

        # Adjusting the weight for intra and inter modal inputs
        intra_weight = (1 - inter_weight) / num_intra
        inter_weight = inter_weight / num_inter

        # Intra: first num_intra elements, Inter: remaining num_inter elements
        intra_part = inputs[:num_intra]
        inter_part = inputs[num_intra:]

        # Weighted mean calculation
        weighted_intra = intra_weight * torch.sum(intra_part, dim=0)
        weighted_inter = inter_weight * torch.sum(inter_part, dim=0)

        # Final output combines intra and inter with adjusted weights
        x = weighted_intra + weighted_inter

        return x


    def gate_loss(self):
        g_loss = []
        for mn, mm in self.named_modules():
            # print(mn)
            if hasattr(mm, 'all_gates'):
                for i in range(len(mm.all_gates)):
                    i_loss = mm.all_gates[f'{i}'].get_loss()
                    if i_loss is None:
                        print(f"[WARN] The gate loss if {mn}, modality: {i} is emtpy, check weather call <get_loss> twice.")
                    else:
                        g_loss.append(i_loss)
        return sum(g_loss)



class FusionLayer(nn.Module):
    def __init__(self, num_modalities, num_patches, hidden_dim, output_dim, num_layers, num_layers_pred, num_experts, num_routers, top_k, num_heads=2, dropout=0.5, mlp_sparse=False):
        super(FusionLayer, self).__init__()
        layers = []
        layers.append(TransformerEncoderLayer(num_experts, num_routers, hidden_dim, num_head=num_heads, dropout=dropout, hidden_times=2, mlp_sparse=mlp_sparse, top_k=top_k))
        for j in range(num_layers - 1):
            tmp = (mlp_sparse) & (j % 2 == 1)
            layers.append(TransformerEncoderLayer(num_experts, num_routers, hidden_dim, num_head=num_heads, dropout=dropout, hidden_times=2, mlp_sparse=tmp, top_k=top_k))
        layers.append(MLP(hidden_dim*num_modalities, hidden_dim, output_dim, num_layers_pred, activation=nn.ReLU(), dropout=0.5))
        
        self.network = nn.Sequential(*layers)
        self.pos_embed = nn.Parameter(torch.zeros(1, np.sum([num_patches]*num_modalities), hidden_dim))

    def forward(self, *inputs):
        chunk_size = [input.shape[1] for input in inputs]
        x = torch.cat(inputs, dim=1)
        if self.pos_embed != None:
            x += self.pos_embed

        x = torch.split(x, chunk_size, dim=1)

        for i in range(len(self.network) - 1):
            x = self.network[i](x)
        x = [item.mean(dim=1) for item in x]
        x = torch.cat(x, dim=1)
        x = self.network[-1](x)
        return x

    def gate_loss(self):
        g_loss = []
        for mn, mm in self.named_modules():
            # print(mn)
            if hasattr(mm, 'all_gates'):
                for i in range(len(mm.all_gates)):
                    i_loss = mm.all_gates[f'{i}'].get_loss()
                    if i_loss is None:
                        print(f"[WARN] The gate loss if {mn}, modality: {i} is emtpy, check weather call <get_loss> twice.")
                    else:
                        g_loss.append(i_loss)
        return sum(g_loss)


class PatchEmbeddings(nn.Module):
    """
    Image to Patch Embedding.

    """
    def __init__(self, feature_size, num_patches, embed_dim, dropout=0.25):
        super().__init__()
        patch_size = math.ceil(feature_size / num_patches)
        pad_size = num_patches*patch_size - feature_size
        self.pad_size = pad_size
        self.num_patches = num_patches
        self.feature_size = feature_size
        self.patch_size = patch_size
        self.projection = nn.Linear(patch_size, embed_dim)

    def forward(self, x):
        x = F.pad(x, (0, self.pad_size)).view(x.shape[0], self.num_patches, self.patch_size)
        # x = F.normalize(x, dim=-1)
        x = self.projection(x)
        return x



class TransformerEncoderLayer(nn.Module):
    def __init__(self, 
                num_experts,
                num_routers,
                d_model, 
                num_head, 
                dropout=0.1, 
                activation=nn.GELU, 
                hidden_times=2, 
                mlp_sparse = False, 
                self_attn = True,
                top_k=2,
                **kwargs) -> None:
        super(TransformerEncoderLayer, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation()
        self.attn = Attention(
            d_model, num_heads=num_head, qkv_bias=False, attn_drop=dropout, proj_drop=dropout)
        
        self.mlp_sparse = mlp_sparse
        self.self_attn = self_attn

        if self.mlp_sparse:
            self.mlp = FMoETransformerMLP(num_expert=num_experts, n_router=num_routers, d_model=d_model, d_hidden=d_model * hidden_times, activation=nn.GELU(), top_k=top_k, **kwargs)
        else:
            self.mlp = MLP(input_dim=d_model, hidden_dim=d_model * hidden_times, output_dim=d_model, num_layers=2, activation=nn.GELU(), dropout=dropout)

    def forward(self, x, expert_index=None):
        if self.self_attn:
            if expert_index:
                x = self.attn(x, x)
                x = x + self.dropout1(x)
                x = self.mlp(self.norm2(x), expert_index)
                return x
            
            else:
                chunk_size = [item.shape[1] for item in x]
                x = self.norm1(torch.cat(x, dim=1))
                kv = x
                x = self.attn(x, kv)
                x = x + self.dropout1(x)
                x = torch.split(x, chunk_size, dim=1)
                x = [item for item in x]

                if self.mlp_sparse:
                    for i in range(len(chunk_size)):
                        x[i] = x[i] + self.dropout2(self.mlp(self.norm2(x[i]), expert_index))
                else:
                    for i in range(len(chunk_size)):
                        x[i] = x[i] + self.dropout2(self.mlp(self.norm2(x[i])))
        else:
            chunk_size = [item.shape[1] for item in x]
            x = [item for item in x]
            for i in range(len(chunk_size)):
                other_m = [x[j] for j in range(len(chunk_size)) if j != i]
                other_m = torch.cat([x[i], *other_m], dim=1)
                x[i] = self.attn(x[i], other_m)
            x = [x[i]+self.dropout1(x[i]) for i in range(len(chunk_size))]
            if self.mlp_sparse:
                for i in range(len(chunk_size)):
                    x[i] = x[i] + self.dropout2(self.mlp(self.norm2(x[i]), expert_index))
            else:
                for i in range(len(chunk_size)):
                    x[i] = x[i] + self.dropout2(self.mlp(self.norm2(x[i])))
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.head_dim = head_dim
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(head_dim * self.num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x, kv, attn_mask=None):
        # attn_mask: (B, N+1, N+1) input-dependent
        eps = 1e-6

        Bx, Nx, Cx = x.shape
        B, N, C = kv.shape
        q = self.q(x).reshape(Bx, Nx, self.num_heads, Cx//self.num_heads)
        q = q.permute(0, 2, 1, 3)
        kv = self.kv(kv)
        kv = kv.reshape(B, N, 2, self.num_heads, C // self.num_heads)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N+1, C/H) @ (B, H, C/H, N+1) -> (B, H, N+1, N+1)

        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(Bx, Nx, -1)  # (B, H, N+1, N+1) * (B, H, N+1, C/H) -> (B, H, N+1, C/H) -> (B, N+1, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
