import numpy as np
import torch
from sklearn.mixture import GaussianMixture


class GaussianPrototypes:
    """隐空间 k 个全局高斯原型：离线用 GMM 拟合后冻结。

    参数形状：mu/logvar/pi 分别为 [k, D] / [k, D] / [k]，D = 隐空间通道数（VAE latent C=4）。
    推理/训练时对 z0（[B, C, H, W]）算软分配 γ_k、加权均值 z_tar、加权方差 d。
    """

    def __init__(self, n_components, reg_covar=1e-6, random_state=0):
        self.n_components = n_components
        self.reg_covar = reg_covar
        self.random_state = random_state
        self.mu = None       # [k, D]
        self.logvar = None   # [k, D]
        self.pi = None       # [k]

    @property
    def var(self):
        return self.logvar.exp()   # [k, D]

    # ---------- 离线拟合（sklearn GMM） ----------
    def fit(self, latents):
        """latents: [N, D] 正常样本隐空间特征（已拉平、已缩放）。"""
        if isinstance(latents, torch.Tensor):
            latents = latents.detach().cpu().numpy()
        gm = GaussianMixture(
            n_components=self.n_components,
            covariance_type='diag',        # 对角协方差
            reg_covar=self.reg_covar,      # 收缩正则，保证 Σ 可逆
            init_params='kmeans',          # k-means++ 初始化 μ
            max_iter=200,
            random_state=self.random_state,
        )
        gm.fit(latents.astype(np.float32))
        self.mu = torch.from_numpy(gm.means_.astype(np.float32))                       # [k, D]
        self.logvar = torch.from_numpy(np.log(gm.covariances_ + 1e-12).astype(np.float32))  # [k, D]
        self.pi = torch.from_numpy(gm.weights_.astype(np.float32))                     # [k]

    # ---------- 存取 ----------
    def save(self, path):
        torch.save({'mu': self.mu, 'logvar': self.logvar, 'pi': self.pi,
                    'n_components': self.n_components}, path)

    def load(self, path, device='cuda'):
        device = torch.device(device)            # int -> cuda:N；str -> 对应设备；torch.device 原样
        ckpt = torch.load(path, map_location=device)
        self.mu = ckpt['mu'].to(device)
        self.logvar = ckpt['logvar'].to(device)
        self.pi = ckpt['pi'].to(device)
        self.n_components = ckpt['n_components']
        return self

    # ---------- 推理/训练：软分配 ----------
    def responsibilities(self, z):
        """z: [B, C, H, W]。返回 γ_k [B, H, W, k]，每位置 Σ_k γ_k = 1。"""
        zt = z.permute(0, 2, 3, 1)                # [B, H, W, D]
        mu = self.mu.to(z.device)
        logvar = self.logvar.to(z.device)
        var = logvar.exp()                        # [k, D]
        pi = self.pi.to(z.device)
        diff = zt.unsqueeze(-2) - mu              # [B, H, W, k, D]
        log_n = -0.5 * (diff * diff / var).sum(-1) - 0.5 * logvar.sum(-1) + pi.log()  # [B, H, W, k]
        log_n = log_n - torch.logsumexp(log_n, dim=-1, keepdim=True)
        return log_n.exp()                        # [B, H, W, k]

    def weighted_mean(self, z):
        """z_tar = Σ_k γ_k · μ_k，返回 [B, C, H, W]。"""
        gamma = self.responsibilities(z)                 # [B, H, W, k]
        z_tar = (gamma.unsqueeze(-1) * self.mu.to(z.device)).sum(-2)  # [B, H, W, D]
        return z_tar.permute(0, 3, 1, 2)                 # [B, C, H, W]

    def weighted_variance(self, z):
        """d = Σ_k γ_k · σ²_k，返回 [B, C, H, W]（逐维，各向异性）。"""
        gamma = self.responsibilities(z)                 # [B, H, W, k]
        var = self.logvar.to(z.device).exp()             # [k, D]
        d = (gamma.unsqueeze(-1) * var).sum(-2)          # [B, H, W, D]
        return d.permute(0, 3, 1, 2)                     # [B, C, H, W]
