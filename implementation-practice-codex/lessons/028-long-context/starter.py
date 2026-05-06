"""Long Context Techniques
========================
긴 시퀀스를 효율적으로 처리하는 방법들.

문제: 표준 attention은 O(S^2) 메모리/연산 → S가 길면 불가능
  S=2048:   4M attention entries
  S=128K:  16B attention entries (4000x 증가!)

해결 방법들:
  1. RoPE + NTK-aware scaling (위치 인코딩 외삽)
  2. Sliding Window Attention (Mistral)
  3. Ring Attention (시퀀스를 GPU에 분산)
  4. KV Cache 최적화 (GQA, MQA, quantized KV cache)

Practice starter
================
Fill the TODO stubs in this file before opening solution.py.
The imports, constants, and main guard are preserved so you can run the same demo after each implementation pass."""
import torch
import torch.nn.functional as F
import math

def rope_scaling_demo():
    """RoPE 외삽 문제와 해결법.

문제: RoPE는 학습 시 본 위치까지만 잘 동작.
      학습: 4K → 추론: 32K 하면 성능 저하.

해결 1: Position Interpolation (Linear Scaling)
  theta'_i = theta_i / scale_factor
  → 위치를 압축해서 학습 범위 안에 매핑
  → 단점: 가까운 토큰의 구별력 감소

해결 2: NTK-aware Scaling (YaRN 등)
  base' = base * scale_factor^(dim/(dim-2))
  → 고주파(가까운 위치)는 유지, 저주파(먼 위치)만 외삽
  → 더 좋은 성능

해결 3: Dynamic NTK
  seq_len이 학습 길이를 넘으면 동적으로 base 조정"""
    raise NotImplementedError('TODO: implement rope_scaling_demo; compare with solution.py only after trying.')

def sliding_window_attention(Q, K, V, window_size):
    """Sliding Window Attention (Mistral, Longformer 등).

각 토큰이 앞의 window_size개 토큰만 attend.
메모리: O(S * W) instead of O(S^2)

여러 layer를 쌓으면 receptive field가 넓어짐:
  Layer 1: 각 토큰이 W개 토큰 봄
  Layer 2: 각 토큰이 2W개 토큰 봄 (layer 1의 W 토큰이 각각 W개를 봤으므로)
  Layer L: 각 토큰이 L*W개 토큰 봄"""
    raise NotImplementedError('TODO: implement sliding_window_attention; compare with solution.py only after trying.')

def simulate_ring_attention():
    """Ring Attention의 chunk별 처리를 시뮬레이션."""
    raise NotImplementedError('TODO: implement simulate_ring_attention; compare with solution.py only after trying.')

def kv_cache_analysis():
    """KV cache 메모리 분석."""
    raise NotImplementedError('TODO: implement kv_cache_analysis; compare with solution.py only after trying.')
if __name__ == '__main__':
    rope_scaling_demo()
    print('\n' + '=' * 60)
    print('Sliding Window Attention Demo')
    print('=' * 60)
    torch.manual_seed(42)
    Q = torch.randn(1, 8, 16)
    K = torch.randn(1, 8, 16)
    V = torch.randn(1, 8, 16)
    out, attn = sliding_window_attention(Q, K, V, window_size=3)
    print(f'  Input: (1, 8, 16), Window: 3')
    print(f'  Attention pattern (token 5 attends to):')
    nonzero = (attn[0, 5] > 0.01).nonzero().flatten().tolist()
    print(f'    tokens {nonzero} (window_size=3 → positions 2-5)')
    simulate_ring_attention()
    kv_cache_analysis()
