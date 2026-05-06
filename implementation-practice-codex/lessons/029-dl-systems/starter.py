"""DL Systems Concepts
=====================
대규모 모델 학습에 필요한 시스템 레벨 지식.

GPU, 네트워크, throughput 분석 등.

Practice starter
================
Fill the TODO stubs in this file before opening solution.py.
The imports, constants, and main guard are preserved so you can run the same demo after each implementation pass."""
import torch

def gpu_specs():
    """주요 GPU 스펙 비교 및 bottleneck 분석."""
    raise NotImplementedError('TODO: implement gpu_specs; compare with solution.py only after trying.')

def interconnect_specs():
    """GPU 간 통신 대역폭."""
    raise NotImplementedError('TODO: implement interconnect_specs; compare with solution.py only after trying.')

def throughput_analysis():
    """Model FLOPS Utilization (MFU) 계산."""
    raise NotImplementedError('TODO: implement throughput_analysis; compare with solution.py only after trying.')

def training_cost_estimate():
    """학습 시간과 비용 추정."""
    raise NotImplementedError('TODO: implement training_cost_estimate; compare with solution.py only after trying.')
if __name__ == '__main__':
    gpu_specs()
    interconnect_specs()
    throughput_analysis()
    training_cost_estimate()
