import torch
import argparse
import matplotlib.pyplot as plt
from models.positional_embedding import RFFT2VEmbedding, get_timestep_embedding

def parse_args():
    parser = argparse.ArgumentParser(description="RFF-T2V 임베딩 시각화 테스트")
    parser.add_argument("--embed_dim", type=int, default=256, help="임베딩 차원")
    parser.add_argument("--rff_dim", type=int, default=64, help="RFF 차원 (최종 2*rff_dim)")
    parser.add_argument("--sigma", type=float, default=1.0, help="RFF 주파수 스케일")
    parser.add_argument("--trainable", action="store_true", help="학습 가능한 RFF 사용")
    return parser.parse_args()

def visualize_embeddings():
    args = parse_args()
    
    # 임베딩 생성
    rff_t2v = RFFT2VEmbedding(
        embed_dim=args.embed_dim,
        rff_dim=args.rff_dim,
        sigma=args.sigma,
        trainable_rff=args.trainable
    )
    
    # 테스트용 시간 값 생성 (0~1 정규화된 값)
    times = torch.linspace(0, 1, 100)
    
    # 두 가지 임베딩 방식 비교
    rff_t2v_emb = rff_t2v(times)
    sincos_emb = get_timestep_embedding(times, args.embed_dim)
    
    # 임베딩 통계 출력
    print(f"RFF-T2V 임베딩 차원: {rff_t2v_emb.shape}")
    print(f"RFF-T2V 임베딩 통계: 평균={rff_t2v_emb.mean():.4f}, 표준편차={rff_t2v_emb.std():.4f}")
    print(f"SinCos 임베딩 통계: 평균={sincos_emb.mean():.4f}, 표준편차={sincos_emb.std():.4f}")
    
    # 시간 임베딩 시각화 (각 열은 다른 시간 차원의 값)
    plt.figure(figsize=(20, 10))
    
    # RFF-T2V 임베딩
    plt.subplot(2, 1, 1)
    plt.imshow(rff_t2v_emb.detach().numpy(), aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title('RFF-T2V Embedding')
    plt.xlabel('임베딩 차원')
    plt.ylabel('시간 스텝')
    
    # 기존 SinCos 임베딩
    plt.subplot(2, 1, 2)
    plt.imshow(sincos_emb.detach().numpy(), aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title('Original SinCos Embedding')
    plt.xlabel('임베딩 차원')
    plt.ylabel('시간 스텝')
    
    plt.tight_layout()
    plt.savefig('embedding_comparison.png')
    plt.close()
    
    # 주파수 분석을 위한 FFT
    plt.figure(figsize=(20, 10))
    
    # 차원별 응답 시각화 (처음 16개 차원만)
    dim_to_plot = min(16, args.embed_dim)
    for i in range(dim_to_plot):
        plt.subplot(4, 4, i+1)
        plt.plot(times.numpy(), rff_t2v_emb[:, i].detach().numpy(), 'r-', label='RFF-T2V')
        plt.plot(times.numpy(), sincos_emb[:, i].detach().numpy(), 'b--', label='SinCos')
        if i == 0:
            plt.legend()
        plt.title(f'차원 {i}')
    
    plt.tight_layout()
    plt.savefig('embedding_responses.png')
    plt.close()
    
    print("시각화 완료. embedding_comparison.png와 embedding_responses.png 파일을 확인하세요.")
    
    # 그라디언트 확인
    t = torch.tensor([0.5], requires_grad=True)
    emb = rff_t2v(t)
    loss = emb.sum()
    loss.backward()
    
    print(f"그라디언트 테스트: t.grad = {t.grad.item():.6f}")
    if args.trainable:
        print(f"RFF 주파수 그라디언트: {rff_t2v.freq.grad.abs().mean().item():.6f}")
        print(f"T2V 가중치 그라디언트: {rff_t2v.w.grad.abs().mean().item():.6f}")
        print(f"T2V 바이어스 그라디언트: {rff_t2v.b.grad.abs().mean().item():.6f}")

if __name__ == "__main__":
    visualize_embeddings() 