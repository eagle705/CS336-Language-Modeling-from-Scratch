"""
MLP Baseline (Single GPU)
==========================
TP/PP를 적용하기 전의 기본 MLP 구현.
이 코드를 기준으로 parallelism 적용 전후를 비교.
"""

import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    """
    Simple MLP: Linear → GELU → Linear → Dropout

    Transformer의 FFN block과 동일한 구조.
    hidden_dim은 보통 4 * embed_dim (GPT-2 style).
    """

    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        x = self.fc1(x)        # (batch, seq_len, hidden_dim)
        x = self.act(x)        # (batch, seq_len, hidden_dim)
        x = self.fc2(x)        # (batch, seq_len, embed_dim)
        x = self.dropout(x)
        return x


class SimpleTransformerMLP(nn.Module):
    """
    Stacked MLP layers (Transformer FFN blocks) for parallelism demo.

    실제 Transformer에서는 Attention + FFN이지만,
    parallelism 설명을 위해 FFN(MLP) block만 N개 쌓은 모델.
    """

    def __init__(self, num_layers, embed_dim, hidden_dim, vocab_size, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([
            MLPBlock(embed_dim, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, input_ids):
        # input_ids: (batch, seq_len)
        x = self.embedding(input_ids)  # (batch, seq_len, embed_dim)

        for layer in self.layers:
            x = x + layer(x)  # residual connection

        x = self.ln(x)
        logits = self.head(x)  # (batch, seq_len, vocab_size)
        return logits


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    # 작은 모델 예시 (인터뷰용 데모 사이즈)
    config = dict(
        num_layers=4,
        embed_dim=256,
        hidden_dim=1024,  # 4x embed_dim
        vocab_size=1000,
    )

    model = SimpleTransformerMLP(**config)
    print(f"Model config: {config}")
    print(f"Total params: {count_params(model):,}")

    # Forward pass test
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
    logits = model(input_ids)
    print(f"Input shape:  {input_ids.shape}")
    print(f"Output shape: {logits.shape}")

    # Loss computation
    targets = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
    loss = nn.functional.cross_entropy(logits.view(-1, config["vocab_size"]), targets.view(-1))
    loss.backward()
    print(f"Loss: {loss.item():.4f}")
    print(f"Gradient norm (fc1 layer0): {model.layers[0].fc1.weight.grad.norm():.4f}")
