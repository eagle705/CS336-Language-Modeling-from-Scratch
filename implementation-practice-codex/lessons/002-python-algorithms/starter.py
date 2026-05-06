"""Python Algorithms for ML Interviews
======================================
ML 인터뷰에서 나올 수 있는 알고리즘/자료구조 문제.
ML 특화 + 일반 코딩 인터뷰 핵심 패턴.

Practice starter
================
Fill the TODO stubs in this file before opening solution.py.
The imports, constants, and main guard are preserved so you can run the same demo after each implementation pass."""

def topk_without_sort(arr, k):
    """Top-K 원소를 O(N) 평균으로 찾기 (Quickselect).
argsort(O(NlogN)) 대신 사용. 대규모 벡터에서 top-k 추출 시 유용."""
    raise NotImplementedError('TODO: implement topk_without_sort; compare with solution.py only after trying.')

def weighted_reservoir_sampling(stream, k, weights):
    """가중치 기반 reservoir sampling.
데이터 스트림에서 가중치에 비례하여 k개 샘플 추출.
데이터 로딩에서 class imbalance 처리에 사용."""
    raise NotImplementedError('TODO: implement weighted_reservoir_sampling; compare with solution.py only after trying.')

def softmax(logits):
    """수치적으로 안정한 softmax 구현.
max를 빼서 overflow 방지 (결과는 동일: exp(x-c)/sum(exp(x-c)) = exp(x)/sum(exp(x)))."""
    raise NotImplementedError('TODO: implement softmax; compare with solution.py only after trying.')

def cross_entropy_loss(probs, target_idx):
    """Cross-entropy: -log(p[target])"""
    raise NotImplementedError('TODO: implement cross_entropy_loss; compare with solution.py only after trying.')

def beam_search(score_fn, vocab_size, beam_width=3, max_len=5):
    """Beam search decoding (간소화 버전).
각 step에서 beam_width개의 후보를 유지.

score_fn(sequence) → log_probs for next token (vocab_size,)"""
    raise NotImplementedError('TODO: implement beam_search; compare with solution.py only after trying.')

def binary_search(arr, target):
    """이진 탐색. O(log N). LR schedule의 step 찾기 등에 활용."""
    raise NotImplementedError('TODO: implement binary_search; compare with solution.py only after trying.')

def merge_sort(arr):
    """Merge sort. O(NlogN). 안정 정렬."""
    raise NotImplementedError('TODO: implement merge_sort; compare with solution.py only after trying.')

class LRUCache:
    """LRU Cache. KV cache eviction 정책에 사용.
O(1) get/put with OrderedDict."""

    def __init__(self, capacity):
        raise NotImplementedError('TODO: implement LRUCache.__init__; compare with solution.py only after trying.')

    def get(self, key):
        raise NotImplementedError('TODO: implement LRUCache.get; compare with solution.py only after trying.')

    def put(self, key, value):
        raise NotImplementedError('TODO: implement LRUCache.put; compare with solution.py only after trying.')

class Trie:
    """Trie (prefix tree). 토크나이저의 vocab lookup에 사용.
O(L) insert/search (L = 문자열 길이)."""

    def __init__(self):
        raise NotImplementedError('TODO: implement Trie.__init__; compare with solution.py only after trying.')

    def insert(self, word):
        raise NotImplementedError('TODO: implement Trie.insert; compare with solution.py only after trying.')

    def search(self, word):
        raise NotImplementedError('TODO: implement Trie.search; compare with solution.py only after trying.')

    def starts_with(self, prefix):
        raise NotImplementedError('TODO: implement Trie.starts_with; compare with solution.py only after trying.')

def topological_sort(graph):
    """위상 정렬 (Kahn's algorithm).
Autograd의 backward에서 computation graph 순회에 사용.

graph: {node: [dependencies]}"""
    raise NotImplementedError('TODO: implement topological_sort; compare with solution.py only after trying.')

def demo():
    raise NotImplementedError('TODO: implement demo; compare with solution.py only after trying.')
if __name__ == '__main__':
    demo()
