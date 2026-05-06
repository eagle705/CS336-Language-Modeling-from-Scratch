"""
Python Algorithms for ML Interviews
======================================
ML 인터뷰에서 나올 수 있는 알고리즘/자료구조 문제.
ML 특화 + 일반 코딩 인터뷰 핵심 패턴.
"""


# ============================================================
# Part 1: ML 특화 알고리즘
# ============================================================

def topk_without_sort(arr, k):
    """
    Top-K 원소를 O(N) 평균으로 찾기 (Quickselect).
    argsort(O(NlogN)) 대신 사용. 대규모 벡터에서 top-k 추출 시 유용.
    """
    import random

    def quickselect(nums, k):
        if len(nums) <= 1:
            return nums
        pivot = random.choice(nums)
        right = [x for x in nums if x > pivot]   # pivot보다 큰 것
        mid = [x for x in nums if x == pivot]
        left = [x for x in nums if x < pivot]

        if k <= len(right):
            return quickselect(right, k)
        elif k <= len(right) + len(mid):
            return mid[:k - len(right)]  # pivot이 top-k에 포함
        else:
            return right + mid + quickselect(left, k - len(right) - len(mid))

    return quickselect(arr, k)


def weighted_reservoir_sampling(stream, k, weights):
    """
    가중치 기반 reservoir sampling.
    데이터 스트림에서 가중치에 비례하여 k개 샘플 추출.
    데이터 로딩에서 class imbalance 처리에 사용.
    """
    import random
    import math
    reservoir = []

    for i, (item, w) in enumerate(zip(stream, weights)):
        # key = random^(1/w) → 가중치 높을수록 key가 클 확률 높음
        key = random.random() ** (1.0 / w) if w > 0 else 0
        if len(reservoir) < k:
            reservoir.append((key, item))
            reservoir.sort()
        elif key > reservoir[0][0]:
            reservoir[0] = (key, item)
            reservoir.sort()

    return [item for _, item in reservoir]


def softmax(logits):
    """
    수치적으로 안정한 softmax 구현.
    max를 빼서 overflow 방지 (결과는 동일: exp(x-c)/sum(exp(x-c)) = exp(x)/sum(exp(x))).
    """
    import math
    max_val = max(logits)
    exps = [math.exp(x - max_val) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


def cross_entropy_loss(probs, target_idx):
    """Cross-entropy: -log(p[target])"""
    import math
    return -math.log(probs[target_idx] + 1e-10)


def beam_search(score_fn, vocab_size, beam_width=3, max_len=5):
    """
    Beam search decoding (간소화 버전).
    각 step에서 beam_width개의 후보를 유지.

    score_fn(sequence) → log_probs for next token (vocab_size,)
    """
    # 초기: 빈 시퀀스
    beams = [(0.0, [])]  # (log_prob, token_sequence)

    for _ in range(max_len):
        candidates = []
        for score, seq in beams:
            log_probs = score_fn(seq)  # (vocab_size,)
            for token_id in range(vocab_size):
                new_score = score + log_probs[token_id]
                candidates.append((new_score, seq + [token_id]))

        # top-k 후보만 유지
        candidates.sort(reverse=True, key=lambda x: x[0])
        beams = candidates[:beam_width]

    return beams[0][1]  # best sequence


# ============================================================
# Part 2: 일반 코딩 인터뷰 패턴
# ============================================================

def binary_search(arr, target):
    """이진 탐색. O(log N). LR schedule의 step 찾기 등에 활용."""
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def merge_sort(arr):
    """Merge sort. O(NlogN). 안정 정렬."""
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


class LRUCache:
    """
    LRU Cache. KV cache eviction 정책에 사용.
    O(1) get/put with OrderedDict.
    """
    from collections import OrderedDict

    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = self.OrderedDict()

    def get(self, key):
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)  # 최근 사용으로 표시
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)  # 가장 오래된 항목 제거


class Trie:
    """
    Trie (prefix tree). 토크나이저의 vocab lookup에 사용.
    O(L) insert/search (L = 문자열 길이).
    """

    def __init__(self):
        self.children = {}
        self.is_end = False

    def insert(self, word):
        node = self
        for ch in word:
            if ch not in node.children:
                node.children[ch] = Trie()
            node = node.children[ch]
        node.is_end = True

    def search(self, word):
        node = self
        for ch in word:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return node.is_end

    def starts_with(self, prefix):
        node = self
        for ch in prefix:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return True


def topological_sort(graph):
    """
    위상 정렬 (Kahn's algorithm).
    Autograd의 backward에서 computation graph 순회에 사용.

    graph: {node: [dependencies]}
    """
    from collections import deque

    in_degree = {node: 0 for node in graph}
    for node in graph:
        for dep in graph[node]:
            in_degree[dep] = in_degree.get(dep, 0) + 1

    queue = deque([n for n in in_degree if in_degree[n] == 0])
    result = []

    while queue:
        node = queue.popleft()
        result.append(node)
        for dep in graph.get(node, []):
            in_degree[dep] -= 1
            if in_degree[dep] == 0:
                queue.append(dep)

    return result


# ============================================================
# Part 3: Demo
# ============================================================

def demo():
    print("=" * 60)
    print("Algorithm Demos")
    print("=" * 60)

    # Top-K
    arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    print(f"\n  Top-3 of {arr}: {sorted(topk_without_sort(arr, 3), reverse=True)}")

    # Softmax
    logits = [2.0, 1.0, 0.5]
    probs = softmax(logits)
    print(f"  Softmax({logits}) = [{', '.join(f'{p:.4f}' for p in probs)}]")
    print(f"  CE loss (target=0) = {cross_entropy_loss(probs, 0):.4f}")

    # LRU Cache
    cache = LRUCache(2)
    cache.put(1, "a")
    cache.put(2, "b")
    print(f"\n  LRU Cache: get(1)={cache.get(1)}")
    cache.put(3, "c")  # evicts key 2
    print(f"  After put(3): get(2)={cache.get(2)} (evicted)")

    # Trie
    trie = Trie()
    for word in ["the", "there", "their", "them"]:
        trie.insert(word)
    print(f"\n  Trie: search('the')={trie.search('the')}")
    print(f"  Trie: search('they')={trie.search('they')}")
    print(f"  Trie: starts_with('the')={trie.starts_with('the')}")

    # Topological sort (autograd graph)
    graph = {"loss": ["z"], "z": ["y1", "y2"], "y1": ["x"], "y2": ["x"], "x": []}
    print(f"\n  Topological sort (autograd): {topological_sort(graph)}")

    # Binary search
    arr = [1, 3, 5, 7, 9, 11]
    print(f"  Binary search {arr}, target=7: index={binary_search(arr, 7)}")

    # Merge sort
    print(f"  Merge sort [5,3,8,1,2]: {merge_sort([5, 3, 8, 1, 2])}")


if __name__ == "__main__":
    demo()
