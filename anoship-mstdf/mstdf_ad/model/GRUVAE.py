import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUVAE(nn.Module):
    """
    GRU-based Variational Autoencoder for multivariate time series anomaly detection.
    Input shape: (batch_size, seq_len, feature_dim)
    Output: recon, recon_loss
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=2, dropout=0.1):
        super(GRUVAE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        # Encoder: GRU + FC to mean and logvar
        self.gru_encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder: from latent to hidden, then GRU to reconstruct
        self.fc_decoder = nn.Linear(latent_dim, hidden_dim)
        self.gru_decoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc_output = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x):
        # x: (B, T, N)
        _, h_n = self.gru_encoder(x)  # h_n: (num_layers, B, hidden_dim)
        h_last = h_n[-1]  # take last layer: (B, hidden_dim)
        mu = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, seq_len):
        # z: (B, latent_dim)
        h0 = self.fc_decoder(z).unsqueeze(0).repeat(self.num_layers, 1, 1)  # (num_layers, B, hidden_dim)
        
        # Initialize input as zeros
        batch_size = z.size(0)
        decoder_input = torch.zeros(batch_size, seq_len, self.input_dim).to(z.device)
        
        # GRU decoding step-by-step
        output, _ = self.gru_decoder(decoder_input, h0)
        recon = self.fc_output(output)  # (B, T, N)
        return recon
    
    def forward(self, x):
        # x: (B, T, N)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, x.size(1))
        
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        total_loss = recon_loss + 0.001 * kl_loss
        
        return recon, total_loss

# ==================== GRU-AE (论文中的对比方法) ====================
class GRUAE(nn.Module):
    """论文中作为对比的GRU-AE方法（Table 4中对比）"""
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=2, dropout=0.1):
        super(GRUAE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        # Encoder
        self.gru_encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc_encoder = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.fc_decoder = nn.Linear(latent_dim, hidden_dim)
        self.gru_decoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc_output = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x):
        _, h_n = self.gru_encoder(x)
        h_last = h_n[-1]
        z = self.fc_encoder(h_last)
        return z
    
    def decode(self, z, seq_len):
        batch_size = z.size(0)
        h0 = self.fc_decoder(z).unsqueeze(0).repeat(self.num_layers, 1, 1)
        
        decoder_input = torch.zeros(batch_size, seq_len, self.input_dim).to(z.device)
        output, _ = self.gru_decoder(decoder_input, h0)
        recon = self.fc_output(output)
        return recon
    
    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z, x.size(1))
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        return recon, recon_loss


# ==================== AE (基础自编码器) ====================
class BasicAE(nn.Module):
    """基础的自编码器对比方法"""
    def __init__(self, input_dim, hidden_dim, latent_dim, seq_len):
        super(BasicAE, self).__init__()
        
        total_features = input_dim * seq_len
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(total_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, total_features)
        )
        
        self.seq_len = seq_len
        self.input_dim = input_dim
        
    def forward(self, x):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        
        z = self.encoder(x_flat)
        recon_flat = self.decoder(z)
        recon = recon_flat.view(batch_size, self.seq_len, self.input_dim)
        
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        return recon, recon_loss
        

# ==================== GDN (图偏差网络) ====================
class GDN(nn.Module):
    """GDN (Graph Deviation Network) - 图偏差网络"""
    def __init__(self, input_dim, hidden_dim, seq_len):
        super(GDN, self).__init__()
        
        self.seq_len = seq_len
        self.input_dim = input_dim
        
        # GAT-like attention
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=4,
            batch_first=True
        )
        
        # GRU for temporal modeling
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        
        # Output layer
        self.fc_out = nn.Linear(hidden_dim, input_dim)
        
        # Learnable graph structure (邻接矩阵)
        self.graph = nn.Parameter(torch.randn(input_dim, input_dim))
        
    def forward(self, x):
        # 图注意力
        attn_output, _ = self.attention(x, x, x)
        
        # GRU时序建模
        gru_output, _ = self.gru(attn_output)
        
        # 重构
        recon = self.fc_out(gru_output)
        
        # 重构损失
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        
        # 图偏差损失（特征间的偏差）
        batch_mean = torch.mean(x, dim=[0, 1])
        graph_dev = torch.matmul(batch_mean, self.graph)
        graph_loss = torch.mean(graph_dev ** 2)
        
        total_loss = recon_loss + 0.01 * graph_loss
        
        return recon, total_loss


# ==================== DVSAD (深度变分支持异常检测) ====================
class DVSAD(nn.Module):
    """DVSAD (Deep Variational Support Anomaly Detection) - 结合VAE和SVDD"""
    def __init__(self, input_dim, hidden_dim, latent_dim, seq_len):
        super(DVSAD, self).__init__()
        
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # VAE部分
        self.gru_encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        self.fc_decoder = nn.Linear(latent_dim, hidden_dim)
        self.gru_decoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.fc_output = nn.Linear(hidden_dim, input_dim)
        
        # SVDD中心点
        self.center = nn.Parameter(torch.randn(latent_dim))
        
    def encode(self, x):
        _, h_n = self.gru_encoder(x)
        h_last = h_n[-1]
        mu = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, seq_len):
        batch_size = z.size(0)
        h0 = self.fc_decoder(z).unsqueeze(0)
        
        decoder_input = torch.zeros(batch_size, seq_len, self.input_dim).to(z.device)
        output, _ = self.gru_decoder(decoder_input, h0)
        recon = self.fc_output(output)
        return recon
    
    def forward(self, x):
        # VAE部分
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, x.size(1))
        
        # 重构损失
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        
        # KL散度
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # SVDD距离损失
        svdd_distances = torch.sum((z - self.center) ** 2, dim=1)
        svdd_loss = torch.mean(svdd_distances)
        
        # 总损失
        total_loss = recon_loss + 0.001 * kl_loss + 0.1 * svdd_loss
        
        return recon, total_loss