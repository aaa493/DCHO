import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialConvolutionLayer(nn.Module):
    def __init__(self, F_latent):
        super(SpatialConvolutionLayer, self).__init__()

        self.W_query = nn.Linear(F_latent, F_latent)
        self.W_key = nn.Linear(F_latent, F_latent)
        self.W_value = nn.Linear(F_latent, F_latent)

        self.activation = nn.ReLU()

    def normalize_adj(self, A):

        B, T, N, _ = A.shape
        I = torch.eye(N).to(A.device)

        A_hat = A + I[None, None, :, :]
        D = torch.sum(A_hat, dim=-1)  # [B, T, N]
        D_inv_sqrt = torch.pow(D, -0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0

        D1 = D_inv_sqrt.unsqueeze(-1)  # [B, T, N, 1]
        D2 = D_inv_sqrt.unsqueeze(-2)  # [B, T, 1, N]

        A_norm = A_hat * D1 * D2
        return A_norm


    def forward(self, hidden_x, A):
        A_pos = A.abs()

        A_norm = self.normalize_adj(A_pos)

        Q = self.W_query(hidden_x)  # (batch_size, T, N, F_out)
        K = self.W_key(hidden_x)    # (batch_size, T, N, F_out)
        V = self.W_value(hidden_x)  # (batch_size, T, N, F_out)

        Q_reshaped = Q.unsqueeze(3)  # (batch_size, T, N, 1, F_out)
        K_reshaped = K.unsqueeze(2)  # (batch_size, T, 1, N, F_out)

        cos_sim = F.cosine_similarity(Q_reshaped, K_reshaped, dim=-1)  # (batch_size, T, N, N)

        interaction_scores = A_norm * cos_sim
        attention_weights = F.softmax(interaction_scores, dim=-1)

        aggregated_neighbors = torch.matmul(attention_weights, V)

        X_next = self.activation(aggregated_neighbors)

        return X_next

class MultiLayerSpatialConvolutionLayer(nn.Module):
    def __init__(self, F_in, F_latent, spatial_num_layers, dropout=0.1):

        super(MultiLayerSpatialConvolutionLayer, self).__init__()
        self.num_layers = spatial_num_layers
        
        self.input_layer = nn.Linear(F_in, F_latent)

        self.layers = nn.ModuleList([
            SpatialConvolutionLayer(F_latent) for _ in range(self.num_layers)])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(F_latent) for _ in range(self.num_layers)])
        
        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(self.num_layers)])

    def forward(self, X, A):        

        hidden_x = self.input_layer(X)  # (batch_size, T, N, F_latent)

        out = hidden_x

        for layer, norm, drop in zip(self.layers, self.norms, self.dropouts):
            residual = out
            output = layer(out, A)
            output = drop(output)
            out = norm(output + residual)
        return out

class ChebyshevConv(nn.Module):
    def __init__(self, F_latent, K=3):
        super(ChebyshevConv, self).__init__()
        self.K = K
        self.lambda_max = 2
        self.weight = nn.Parameter(torch.Tensor(K, F_latent, F_latent))
        self.activation = nn.ReLU()
        nn.init.xavier_uniform_(self.weight)

    def normalized_laplacian(self, A):
        B, T, N, _ = A.shape
        I = torch.eye(N, device=A.device).unsqueeze(0).unsqueeze(0)
        A_hat = A + I
        D = A_hat.sum(dim=-1, keepdim=True)
        D_inv_sqrt = torch.pow(D, -0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0
        D_inv_sqrt = D_inv_sqrt.expand(-1, -1, -1, A.size(-1))
        L_hat = I - D_inv_sqrt * A_hat * D_inv_sqrt.transpose(-1, -2)
        return L_hat
    
    def forward(self, hidden_x, A):
        batch_size, T, N, F_in = hidden_x.shape

        A_pos = A.abs()
        laplacian = self.normalized_laplacian(A_pos) 
        I = torch.eye(N, device=A.device).unsqueeze(0).unsqueeze(0)
        L_hat = (2 / self.lambda_max) * laplacian - I
        
        Tx_0 = hidden_x
        Tx_1 = torch.matmul(L_hat, hidden_x)
        
        output = torch.matmul(Tx_0, self.weight[0])
        output += torch.matmul(Tx_1, self.weight[1])
 
        for k in range(2, self.K):
            Tx_2 = 2 * torch.matmul(L_hat, Tx_1) - Tx_0
            output += torch.matmul(Tx_2, self.weight[k])
            Tx_0, Tx_1 = Tx_1, Tx_2

        output = self.activation(output)
        return output
    
class MultiLayerChebyshevConv(nn.Module):
    def __init__(self, F_in, F_latent, che_num_layers=2, K=3, dropout=0.1):

        super(MultiLayerChebyshevConv, self).__init__()
        self.num_layers = che_num_layers
        
        self.input_layer = nn.Linear(F_in, F_latent)

        self.layers = nn.ModuleList([
            ChebyshevConv(F_latent, K=K) for _ in range(self.num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(F_latent) for _ in range(self.num_layers)
        ])
        
        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(self.num_layers)])
        
    def forward(self, X, A):
        hidden_x = self.input_layer(X)  # (batch_size, T, N, F_latent)
        out = hidden_x
        for layer, norm, drop in zip(self.layers, self.norms, self.dropouts):
            residual = out
            output = layer(out, A)
            output = drop(output)
            out = norm(output + residual)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model):

        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

    def forward(self, pos):

        device = pos.device
        T = pos.size(0)
        d_model = self.d_model

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32, device=device) *
            (-torch.log(torch.tensor(10000.0, device=device)) / d_model)
        )  # (D/2,)

        pe = torch.zeros(T, d_model, device=device)  # (T, D)
        pe[:, 0::2] = torch.sin(pos.unsqueeze(1) * div_term)  # broadcasting: (T, 1) * (D/2)
        pe[:, 1::2] = torch.cos(pos.unsqueeze(1) * div_term)

        return pe  # (T, D)

class TemporalTransformerLayer(nn.Module):
    def __init__(self, F_latent, num_layers=2, num_heads=4, dropout=0.1):
        super(TemporalTransformerLayer, self).__init__()
        self.F_latent = F_latent

        self.pos_encoder = PositionalEncoding(d_model=F_latent)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=F_latent,
            nhead=num_heads,
            dim_feedforward=4 * F_latent,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, X):
        B, T, N, F = X.shape
        X = X.permute(0, 2, 1, 3).reshape(B * N, T, F)  # (B*N, T, F)

        pos = torch.arange(T, device=X.device)
        pos_encoding = self.pos_encoder(pos).unsqueeze(0)  # (1, T, F)
        X = X + pos_encoding

        X_out = self.transformer(X)
        X_out = X_out.reshape(B, N, T, F).permute(0, 2, 1, 3)  # (B, T, N, F)
        return X_out


class encoderModel(nn.Module):
    def __init__(self, config_en, act_fn=nn.Softplus):
        super(encoderModel, self).__init__()

        self.F_in = config_en['F_in']
        self.F_latent = config_en['F_latent']
        self.N = config_en['n_regions']
        self.spatial_num_layers = config_en['spatial_layers']
        self.che_num_layers = config_en['che_layers']

        self.GnnLayer = MultiLayerSpatialConvolutionLayer(self.F_in, self.F_latent, self.spatial_num_layers)
        self.CheLayer = MultiLayerChebyshevConv(self.F_in, self.F_latent, self.che_num_layers)
        self.TemporalLayer = TemporalTransformerLayer(self.F_latent)

    def forward(self, X, A):

        temp_Z = 0.5 * self.GnnLayer(X, A)  + 0.5 * self.CheLayer(X, A)

        Z_seq = self.TemporalLayer(temp_Z)

        return Z_seq

class Lstmstream(nn.Module):
    def __init__(self, config_pred, dropout=0.1):
        super().__init__()
        self.F_latent = config_pred['F_latent']
        self.N = config_pred['n_regions']
        self.layers = config_pred['layers']

        self.input_dim = self.N * self.F_latent

        self.lstm = nn.LSTM(
            input_size = self.input_dim,
            hidden_size = self.input_dim,
            num_layers = self.layers,
            batch_first = True,
            dropout = dropout
        )

        self.predictor = nn.Linear(self.input_dim, self.input_dim)
        self.norm = nn.LayerNorm(self.input_dim)

    def forward(self, z):
        B, T, N, F = z.shape
        z = z.reshape(B, T, N * F)

        out, _ = self.lstm(z)

        out = self.predictor(out)
        out = self.norm(out)

        out = out.view(B, T, N, F)

        return out


class DualStreamDecoder(nn.Module):
    def __init__(self, config_de, nhead=4, dropout=0.1):
        super().__init__()
        self.F_latent = config_de['F_latent']
        self.N = config_de['n_regions']
        self.num_layers = config_de['layers']

        self.time_embedding = PositionalEncoding(d_model=self.F_latent)
        self.position_embedding = PositionalEncoding(d_model=self.F_latent)

        self.linear_i = nn.Linear(self.F_latent, self.F_latent)
        self.linear_j = nn.Linear(self.F_latent, self.F_latent)
        self.linear_k = nn.Linear(self.F_latent, self.F_latent)
        self.embedding_proj = nn.Linear(3 * self.F_latent, self.F_latent)

        spatial_layer = nn.TransformerEncoderLayer(
            d_model=self.F_latent,
            nhead=nhead,
            dim_feedforward=self.F_latent * 2,
            dropout=dropout,
            batch_first=True
        )
        self.spatial_transformer = nn.TransformerEncoder(spatial_layer, num_layers=self.num_layers)

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=self.F_latent,
            nhead=nhead,
            dim_feedforward=self.F_latent * 2,
            dropout=dropout,
            batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(temporal_layer, num_layers=self.num_layers)

        self.norm_spatial = nn.LayerNorm(self.F_latent)
        self.norm_temporal = nn.LayerNorm(self.F_latent)

        self.output_proj = nn.Sequential(
            nn.Linear(2 * self.F_latent, self.F_latent),
            nn.ReLU(),
            nn.Linear(self.F_latent, 1)
        )

        import itertools
        idx = list(itertools.combinations_with_replacement(range(self.N), 3))  # M = (N+2 choose 3)
        self.register_buffer('idx_i', torch.tensor([i for i, _, _ in idx]))
        self.register_buffer('idx_j', torch.tensor([j for _, j, _ in idx]))
        self.register_buffer('idx_k', torch.tensor([k for _, _, k in idx]))
        self.M = len(idx)

    def forward(self, z):
        B, T, N, F = z.shape
        h_i = self.linear_i(z)
        h_j = self.linear_j(z)
        h_k = self.linear_k(z)

        h_i_expand = h_i[:, :, self.idx_i]  # (B, T, M, F)
        h_j_expand = h_j[:, :, self.idx_j]
        h_k_expand = h_k[:, :, self.idx_k]

        triple_feat = torch.cat([h_i_expand, h_j_expand, h_k_expand], dim=-1)  # (B, T, M, 3F)
        token_feat = self.embedding_proj(triple_feat)  # (B, T, M, F)

        # === Spatial Transformer ===
        spatial_input = token_feat.reshape(B * T, self.M, self.F_latent)  # (B*T, M, F)
        spatial_input = spatial_input + self.position_embedding(torch.arange(self.M, device=z.device)).unsqueeze(0)
        spatial_out = self.spatial_transformer(spatial_input)
        spatial_out = self.norm_spatial(spatial_out)
        spatial_out = spatial_out.reshape(B, T, self.M, self.F_latent)

        # === Temporal Transformer ===
        temporal_input = token_feat.permute(0, 2, 1, 3).reshape(B * self.M, T, self.F_latent)
        temporal_input = temporal_input + self.time_embedding(torch.arange(T, device=z.device)).unsqueeze(0)
        temporal_out = self.temporal_transformer(temporal_input)
        temporal_out = self.norm_temporal(temporal_out)
        temporal_out = temporal_out.reshape(B, self.M, T, self.F_latent).permute(0, 2, 1, 3)  # (B, T, M, F)

        # === Concatenate both outputs ===
        feat_cat = torch.cat([spatial_out, temporal_out], dim=-1)  # (B, T, M, 2F)
        out = self.output_proj(feat_cat).squeeze(-1)  # (B, T, M)

        return out



    