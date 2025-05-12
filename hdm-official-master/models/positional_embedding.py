import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class RFFT2VEmbedding(nn.Module):
    """Random Fourier Feature + Time2Vec 결합 임베딩
    
    Parameters
    ----------
    embed_dim : int
        최종 임베딩 차원
    rff_dim : int, optional
        Random Fourier Feature 차원 (최종 차원은 2*rff_dim), 기본값은 embed_dim의 1/4
    sigma : float, optional
        RFF 주파수 초기화 표준편차, 기본값 1.0
    trainable_rff : bool, optional
        RFF 주파수를 학습 가능하게 설정할지 여부, 기본값 False
    """
    def __init__(self, embed_dim, rff_dim=None, sigma=1.0, trainable_rff=False):
        super().__init__()
        self.embed_dim = embed_dim
        
        # RFF와 T2V 차원 배분
        if rff_dim is None:
            rff_dim = embed_dim // 4  # 기본 RFF 차원 (최종 2*rff_dim)
        
        t2v_dim = embed_dim - 2 * rff_dim
        assert t2v_dim >= 1, f"임베딩 차원 배분 오류: embed_dim={embed_dim}, rff_dim={rff_dim}"
        
        self.rff_dim = rff_dim
        self.t2v_dim = t2v_dim
        
        # Random Fourier Feature 초기화
        freq = torch.randn(rff_dim) * sigma
        self.register_buffer("freq", freq, persistent=True)
        if trainable_rff:
            self.freq = nn.Parameter(freq)  # 학습 가능하게 설정
        
        # Time2Vec 초기화
        self.w = nn.Parameter(torch.randn(t2v_dim))
        self.b = nn.Parameter(torch.zeros(t2v_dim))
    
    def forward(self, t):
        """시간 임베딩 계산
        
        Parameters
        ----------
        t : torch.Tensor
            시간 텐서, 형태 (batch_size,) 또는 (batch_size, 1)
            
        Returns
        -------
        torch.Tensor
            형태 (batch_size, embed_dim)의 시간 임베딩
        """
        if t.dim() == 1:
            t = t.view(-1, 1)  # 차원 보정
        
        # RFF 부분
        proj = t * self.freq  # (batch_size, rff_dim)
        rff_part = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (batch_size, 2*rff_dim)
        
        # T2V 부분
        t2v_part = torch.zeros((t.shape[0], self.t2v_dim), device=t.device)
        
        # 선형 성분 (첫 번째 차원)
        t2v_part[:, 0] = self.w[0] * t.squeeze() + self.b[0]
        
        # 주기적 성분 (나머지 차원)
        if self.t2v_dim > 1:
            t2v_part[:, 1:] = torch.sin(t * self.w[1:] + self.b[1:])
        
        # 결합
        return torch.cat([rff_part, t2v_part], dim=-1)  # (batch_size, embed_dim)


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    """기존 sinusoidal 임베딩 (호환성 유지)"""
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    timesteps = 1000 * timesteps
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb 