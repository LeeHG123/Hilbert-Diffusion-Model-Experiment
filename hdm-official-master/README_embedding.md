# RFF-T2V 시간 임베딩

Hilbert Diffusion Model (HDM)의 Fourier Neural Operator (FNO) Score Network에서 사용할 향상된 시간 임베딩 구현입니다.

## 개요

기존 HDM은 단순한 사인-코사인 임베딩을 사용하지만, 본 구현은 다음 두 임베딩 방식을 결합했습니다:

1. **Random Fourier Features (RFF)**: 가우시안 분포로 초기화된 무작위 주파수를 사용하는 사인-코사인 임베딩
2. **Time2Vec (T2V)**: 선형 및 주기적 컴포넌트를 결합한 시간 임베딩

이 하이브리드 접근법은 다양한 패턴과 주파수를 통해 시간 정보를 보다 풍부하게 인코딩할 수 있습니다.

## 특징

- **더 풍부한 주파수 표현**: 다양한 주파수를 통해 시간 정보 인코딩
- **학습 가능 옵션**: RFF 주파수를 고정하거나 학습 가능하게 설정
- **조정 가능한 구성**: RFF와 T2V 각 부분의 차원과 특성 조정 가능
- **기존 코드와 호환성**: 기존 임베딩 방식과 호환되는 설계

## 사용 방법

### 1. 설정 파일에 파라미터 추가

```yaml
model:
  # 기존 설정...
  time_embedding: "rff_t2v"  # 'sincos'(기존) 또는 'rff_t2v'(새 임베딩)
  rff_dim: 64               # RFF 차원 (실제로는 2*rff_dim 차원 사용)
  rff_sigma: 1.0            # 주파수 초기화 스케일
  trainable_rff: false      # RFF 주파수 학습 여부
```

### 2. 임베딩 객체 사용

```python
from models.positional_embedding import RFFT2VEmbedding

# 임베딩 초기화
embed = RFFT2VEmbedding(
    embed_dim=256,       # 임베딩 총 차원
    rff_dim=64,          # RFF 차원 (2*rff_dim 차원 사용)
    sigma=1.0,           # 주파수 스케일
    trainable_rff=False  # RFF 주파수 학습 여부
)

# 시간값 임베딩
t = torch.tensor([0.1, 0.2, 0.3])  # 배치의 시간값
emb = embed(t)  # 형태: [배치크기, embed_dim]
```

## 테스트 및 시각화

임베딩 테스트 및 시각화를 위해 다음 명령어를 실행하세요:

```bash
python test_embedding.py --embed_dim 256 --rff_dim 64 --sigma 1.0
```

추가 옵션:
- `--trainable`: RFF 주파수를 학습 가능하게 설정

## 임베딩 비교

### 기존 임베딩 (sincos)
- 단일 스케일 로그 주파수 간격을 사용
- 모든 시간 스케일에서 고정된 임베딩 패턴

### 새 임베딩 (RFF-T2V)
- 다양한 주파수 사용으로 시간 정보 더 풍부하게 인코딩
- 선형 시간 컴포넌트 포함 (T2V 부분)
- 데이터셋 특성에 맞춰 학습 가능한 옵션

## 성능 고려사항

- **하이퍼파라미터 탐색**: `rff_dim`과 `sigma` 값을 데이터셋에 맞게 조정하세요
- **메모리 사용량**: 동일한 임베딩 차원에서 기존 방식과 비슷한 메모리 사용
- **학습 안정성**: `trainable_rff=True` 사용 시 학습률 조정이 필요할 수 있음 