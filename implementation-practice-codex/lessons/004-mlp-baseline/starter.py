"""MLP Baseline (Single GPU)
==========================
TP/PP를 적용하기 전의 기본 MLP 구현.
이 코드를 기준으로 parallelism 적용 전후를 비교.

Practice starter
================
Fill the TODO stubs in this file before opening solution.py.
The imports, constants, and main guard are preserved so you can run the same demo after each implementation pass."""
import torch
import torch.nn as nn

class MLPBlock(nn.Module):
    """Simple MLP: Linear → GELU → Linear → Dropout

Transformer의 FFN block과 동일한 구조.
hidden_dim은 보통 4 * embed_dim (GPT-2 style)."""

    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        raise NotImplementedError('TODO: implement MLPBlock.__init__; compare with solution.py only after trying.')

    def forward(self, x):
        raise NotImplementedError('TODO: implement MLPBlock.forward; compare with solution.py only after trying.')

class SimpleTransformerMLP(nn.Module):
    """Stacked MLP layers (Transformer FFN blocks) for parallelism demo.

실제 Transformer에서는 Attention + FFN이지만,
parallelism 설명을 위해 FFN(MLP) block만 N개 쌓은 모델."""

    def __init__(self, num_layers, embed_dim, hidden_dim, vocab_size, dropout=0.1):
        raise NotImplementedError('TODO: implement SimpleTransformerMLP.__init__; compare with solution.py only after trying.')

    def forward(self, input_ids):
        raise NotImplementedError('TODO: implement SimpleTransformerMLP.forward; compare with solution.py only after trying.')

def count_params(model):
    raise NotImplementedError('TODO: implement count_params; compare with solution.py only after trying.')
if __name__ == '__main__':
    config = dict(num_layers=4, embed_dim=256, hidden_dim=1024, vocab_size=1000)
    model = SimpleTransformerMLP(**config)
    print(f'Model config: {config}')
    print(f'Total params: {count_params(model):,}')
    batch_size, seq_len = (2, 16)
    input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    logits = model(input_ids)
    print(f'Input shape:  {input_ids.shape}')
    print(f'Output shape: {logits.shape}')
    targets = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    loss = nn.functional.cross_entropy(logits.view(-1, config['vocab_size']), targets.view(-1))
    loss.backward()
    print(f'Loss: {loss.item():.4f}')
    print(f'Gradient norm (fc1 layer0): {model.layers[0].fc1.weight.grad.norm():.4f}')
