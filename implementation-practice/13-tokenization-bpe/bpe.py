"""
BPE (Byte Pair Encoding) Tokenizer from Scratch
==================================================
텍스트를 subword 토큰으로 분리하는 알고리즘.

왜 subword?
  - Word-level: OOV(미등록 단어) 문제, vocab 너무 큼
  - Char-level: 시퀀스 너무 김, 의미 손실
  - Subword:    둘의 장점 결합 ("unhappiness" → "un" + "happiness")

BPE 알고리즘:
  Training (vocab 구축):
    1. 텍스트를 문자(바이트) 단위로 분리 → 초기 vocab = {모든 바이트}
    2. 가장 빈번한 인접 쌍(pair) 찾기
    3. 그 쌍을 하나의 새 토큰으로 병합
    4. vocab_size에 도달할 때까지 2-3 반복

  Encoding (텍스트 → 토큰):
    1. 텍스트를 문자 단위로 분리
    2. 학습된 병합 규칙을 우선순위 순서대로 적용

  예시:
    텍스트: "low lower lowest"
    초기: ['l','o','w',' ','l','o','w','e','r',' ','l','o','w','e','s','t']
    병합 1: ('l','o') → 'lo'  (가장 빈번한 쌍)
    병합 2: ('lo','w') → 'low'
    병합 3: ('low','e') → 'lowe'
    ...
"""

from collections import Counter, defaultdict


# ============================================================
# Part 1: BPE Trainer
# ============================================================

class BPETokenizer:
    """
    Byte Pair Encoding tokenizer from scratch.
    학습 + 인코딩 + 디코딩 구현.
    """

    def __init__(self):
        self.merges = {}       # (token_a, token_b) → merged_token, 순서대로
        self.vocab = {}        # token → id
        self.id_to_token = {}  # id → token

    def train(self, text, vocab_size=300, verbose=True):
        """
        BPE 학습: 텍스트에서 vocab_size개의 토큰을 구축.

        1. 초기 vocab: 모든 고유 바이트 (최대 256개)
        2. 반복: 가장 빈번한 인접 쌍을 찾아 병합
        """
        if verbose:
            print("=" * 60)
            print("BPE Training")
            print("=" * 60)

        # (1) 텍스트를 바이트 단위로 분리
        #     각 단어를 독립적으로 처리 (GPT-2 스타일: 공백을 Ġ로 표현)
        words = text.split()
        # 각 단어를 문자 튜플 + 빈도로 저장
        word_freqs = Counter(words)
        # 단어를 문자 리스트로 분리
        splits = {word: list(word) for word in word_freqs}

        # (2) 초기 vocab: 모든 고유 문자
        chars = set()
        for word in splits:
            chars.update(splits[word])
        self.vocab = {ch: i for i, ch in enumerate(sorted(chars))}
        next_id = len(self.vocab)

        if verbose:
            print(f"  Initial vocab size: {len(self.vocab)}")
            print(f"  Target vocab size: {vocab_size}")

        # (3) 반복: 병합
        num_merges = vocab_size - len(self.vocab)
        for step in range(num_merges):
            # 모든 인접 쌍의 빈도 세기
            pair_freqs = Counter()
            for word, freq in word_freqs.items():
                tokens = splits[word]
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    pair_freqs[pair] += freq

            if not pair_freqs:
                break

            # 가장 빈번한 쌍 선택
            best_pair = pair_freqs.most_common(1)[0][0]
            best_count = pair_freqs[best_pair]
            merged = best_pair[0] + best_pair[1]

            if verbose and step < 10:
                print(f"  Merge {step}: '{best_pair[0]}' + '{best_pair[1]}' → '{merged}' (count={best_count})")

            # vocab에 추가
            self.merges[best_pair] = merged
            self.vocab[merged] = next_id
            next_id += 1

            # 모든 단어에서 해당 쌍 병합
            for word in splits:
                tokens = splits[word]
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                        new_tokens.append(merged)
                        i += 2  # 쌍을 건너뜀
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                splits[word] = new_tokens

        # id ↔ token 매핑
        self.id_to_token = {v: k for k, v in self.vocab.items()}

        if verbose:
            print(f"  ...")
            print(f"  Final vocab size: {len(self.vocab)}")
            print(f"  Total merges: {len(self.merges)}")

    def encode(self, text):
        """
        텍스트를 토큰 ID 시퀀스로 변환.
        학습된 병합 규칙을 순서대로 적용.
        """
        words = text.split()
        all_ids = []

        for word in words:
            tokens = list(word)

            # 병합 규칙을 순서대로 적용
            for pair, merged in self.merges.items():
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                        new_tokens.append(merged)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens

            # 토큰 → ID
            for token in tokens:
                if token in self.vocab:
                    all_ids.append(self.vocab[token])
                else:
                    # 미등록 토큰: 바이트 단위로 분해
                    for ch in token:
                        all_ids.append(self.vocab.get(ch, 0))

        return all_ids

    def decode(self, ids):
        """토큰 ID 시퀀스를 텍스트로 복원."""
        tokens = [self.id_to_token.get(i, '?') for i in ids]
        # 단어 경계 복원 (간소화 버전)
        return ''.join(tokens)


# ============================================================
# Part 2: Byte-level BPE (GPT-2 style)
# ============================================================
#
# GPT-2/GPT-4 등은 byte-level BPE 사용:
#
# 1. UTF-8 바이트(0-255)를 초기 vocab으로 사용
#    → 어떤 텍스트든 인코딩 가능 (OOV 없음)
#
# 2. 공백 처리: 공백을 특수 문자 'Ġ'(Ġ = 0x120)로 표현
#    "hello world" → "hello", "Ġworld"
#    → 단어 경계를 명시적으로 인코딩
#
# 3. Pre-tokenization: regex로 텍스트를 먼저 분리
#    GPT-2 regex: r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+"""
#    → "I'm happy" → ["I", "'m", " happy"]
#    → 병합이 단어 경계를 넘지 않도록
#
# tiktoken (OpenAI의 빠른 BPE 구현):
#   import tiktoken
#   enc = tiktoken.encoding_for_model("gpt-4")
#   tokens = enc.encode("Hello, world!")  # [9906, 11, 1917, 0]
#   text = enc.decode(tokens)             # "Hello, world!"
#   print(enc.n_vocab)                    # 100277


# ============================================================
# Part 3: SentencePiece (Unigram LM)
# ============================================================
#
# BPE의 대안: Unigram Language Model (SentencePiece)
#
# 차이점:
#   BPE:     bottom-up (작은 단위 → 병합)
#   Unigram: top-down (큰 vocab → 가지치기)
#
# Unigram 알고리즘:
#   1. 큰 후보 vocab으로 시작 (모든 가능한 subword)
#   2. 각 subword의 확률 계산 (Unigram LM)
#   3. 각 subword를 제거했을 때 likelihood 감소량 계산
#   4. 가장 덜 중요한 subword 제거
#   5. 목표 vocab size까지 반복
#
# LLaMA, T5 등에서 SentencePiece 사용.


# ============================================================
# Part 4: Demo
# ============================================================

def demo():
    print("=" * 60)
    print("BPE Tokenizer Demo")
    print("=" * 60)

    text = ("the cat sat on the mat the cat ate the rat "
            "the dog sat on the log the dog ate the frog "
            "the cat and the dog sat on the mat together")

    tokenizer = BPETokenizer()
    tokenizer.train(text, vocab_size=50)

    # 인코딩 테스트
    test_texts = ["the cat", "the dog sat", "mat"]
    print(f"\n  Encoding examples:")
    for t in test_texts:
        ids = tokenizer.encode(t)
        decoded = tokenizer.decode(ids)
        print(f"    '{t}' → {ids} → '{decoded}'")

    # Vocab 분석
    print(f"\n  Vocab (sample):")
    sorted_vocab = sorted(tokenizer.vocab.items(), key=lambda x: x[1])
    for token, id in sorted_vocab[:10]:
        print(f"    {id:3d}: '{token}'")
    print(f"    ...")
    for token, id in sorted_vocab[-5:]:
        print(f"    {id:3d}: '{token}'")

    # 압축률
    char_len = len(text.replace(" ", ""))
    token_len = len(tokenizer.encode(text))
    print(f"\n  Compression:")
    print(f"    Characters: {char_len}")
    print(f"    Tokens:     {token_len}")
    print(f"    Ratio:      {char_len / token_len:.1f} chars/token")


if __name__ == "__main__":
    demo()
