# Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
#               2024 Alibaba Inc (authors: Xiang Lyu)
#               2025 Alibaba Inc (authors: Xiang Lyu, Bofan Zhou)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)
"""Unility functions for Transformer."""

import queue
import random
from typing import List

import numpy as np
import torch

IGNORE_ID = -1


def pad_list(xs: List[torch.Tensor], pad_value: int):
    """Perform padding for the list of tensors.

    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).

    Examples:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])

    """
    max_len = max([len(item) for item in xs])
    batchs = len(xs)
    ndim = xs[0].ndim
    if ndim == 1:
        pad_res = torch.zeros(batchs,
                              max_len,
                              dtype=xs[0].dtype,
                              device=xs[0].device)
    elif ndim == 2:
        pad_res = torch.zeros(batchs,
                              max_len,
                              xs[0].shape[1],
                              dtype=xs[0].dtype,
                              device=xs[0].device)
    elif ndim == 3:
        pad_res = torch.zeros(batchs,
                              max_len,
                              xs[0].shape[1],
                              xs[0].shape[2],
                              dtype=xs[0].dtype,
                              device=xs[0].device)
    else:
        raise ValueError(f"Unsupported ndim: {ndim}")
    pad_res.fill_(pad_value)
    for i in range(batchs):
        pad_res[i, :len(xs[i])] = xs[i]
    return pad_res


def th_accuracy(pad_outputs: torch.Tensor, pad_targets: torch.Tensor,
                ignore_label: int) -> torch.Tensor:
    """Calculate accuracy.

    Args:
        pad_outputs (Tensor): Prediction tensors (B * Lmax, D).
        pad_targets (LongTensor): Target label tensors (B, Lmax).
        ignore_label (int): Ignore label id.

    Returns:
        torch.Tensor: Accuracy value (0.0 - 1.0).

    """
    pad_pred = pad_outputs.view(pad_targets.size(0), pad_targets.size(1),
                                pad_outputs.size(1)).argmax(2)
    mask = pad_targets != ignore_label
    numerator = torch.sum(
        pad_pred.masked_select(mask) == pad_targets.masked_select(mask))
    denominator = torch.sum(mask)
    return (numerator / denominator).detach()


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


# Repetition Aware Sampling in VALL-E 2 Repetition-Aware Sampling(반복 인지 샘플링)
def ras_sampling(weighted_scores, decoded_tokens, sampling, top_p=0.8, top_k=25, win_size=10, tau_r=0.1):
    """
    기본적으로 nucleus_sampling을 따르되, 생성된 토큰이 반복될 경우 이를 감지하고
    random_sampling으로 전환하여 단조로움을 피하는 샘플링 전략.
    """
    # 1. [기본 경로] 먼저 nucleus_sampling을 통해 가장 유력한 다음 토큰을 하나 선택합니다.
    top_ids = nucleus_sampling(weighted_scores, top_p=top_p, top_k=top_k)
    
    # 2. [반복 검사] 선택된 토큰이 바로 직전에 생성된 토큰들 중에 있는지 확인합니다.
    #    - decoded_tokens[-win_size:]: 가장 최근에 생성된 win_size(10)개의 토큰 기록을 가져옵니다.
    #    - ... == top_ids: 기록 중에 방금 선택한 top_ids와 일치하는 것이 있는지 찾습니다.
    #    - .sum().item(): 일치하는 토큰의 개수를 셉니다.
    rep_num = (torch.tensor(decoded_tokens[-win_size:]).to(weighted_scores.device) == top_ids).sum().item()
    
    # 3. [경로 변경] 반복 횟수가 정해진 임계값(win_size * tau_r, 예: 10 * 0.1 = 1)보다 많거나 같으면,
    #    즉, 최근 10개 토큰 안에 이번에 뽑힌 토큰이 '한 번이라도' 있으면...
    if rep_num >= win_size * tau_r:
        # [비상 경로] 1번에서 선택했던 토큰을 버리고, random_sampling을 통해 새 토큰을 다시 뽑아 단조로움을 탈피합니다.
        top_ids = random_sampling(weighted_scores, decoded_tokens, sampling)
        
    # 최종 선택된 토큰을 반환합니다.
    return top_ids


def nucleus_sampling(weighted_scores, top_p=0.8, top_k=25):
    """
    Top-K와 Top-P(Nucleus) 샘플링을 결합하여 다음 토큰 후보군을 만드는 함수.
    Args:
        weighted_scores (torch.Tensor): 모델이 예측한 모든 토큰에 대한 확률 점수(logits).
        top_p (float): 후보군에 포함될 토큰들의 누적 확률 임계값.
        top_k (int): 후보군에 포함될 최대 토큰 개수.
    """
    # 최종 후보군의 확률과 인덱스(토큰 ID)를 저장할 리스트 초기화
    prob, indices = [], []
    # 누적 확률을 계산하기 위한 변수 초기화
    cum_prob = 0.0
    
    # 1. 모델의 확률 점수를 실제 확률(0~1)로 변환하고, 가장 확률이 높은 순서대로 정렬합니다.
    #    sorted_value에는 정렬된 확률값이, sorted_idx에는 해당 토큰의 ID가 들어갑니다.
    sorted_value, sorted_idx = weighted_scores.softmax(dim=0).sort(descending=True, stable=True)

    # 2. 가장 확률이 높은 토큰부터 하나씩 확인하며 후보군을 구성합니다.
    for i in range(len(sorted_idx)):
        # 3. 아래 두 가지 규칙을 '동시에' 만족하는 동안에만 후보군에 토큰을 추가합니다.
        #    - 규칙 1: 지금까지 후보들의 확률 합(cum_prob)이 top_p 값보다 작아야 함.
        #    - 규칙 2: 지금까지 모은 후보의 개수(len(prob))가 top_k 값보다 작아야 함.
        if cum_prob < top_p and len(prob) < top_k:
            # 현재 토큰의 확률을 누적 확률에 더합니다.
            cum_prob += sorted_value[i]
            # 현재 토큰의 확률과 ID를 후보군 리스트에 추가합니다.
            prob.append(sorted_value[i])
            indices.append(sorted_idx[i])
        else:
            # 두 규칙 중 하나라도 만족하지 못하면 후보군 구성을 중단합니다.
            break
            
    # 파이토치 연산을 위해 리스트를 텐서로 변환합니다.
    prob = torch.tensor(prob).to(weighted_scores)
    indices = torch.tensor(indices, dtype=torch.long).to(weighted_scores.device)
    
    # 4. 최종적으로 만들어진 후보군(prob)의 확률에 따라 가중치를 둔 무작위 뽑기를 1번 실행합니다.
    #    뽑힌 인덱스를 사용하여 실제 토큰 ID를 가져와 반환합니다.
    top_ids = indices[prob.multinomial(1, replacement=True)]
    return top_ids


def random_sampling(weighted_scores, decoded_tokens, sampling):
    """
    Top-K, Top-P 필터링 없이 전체 토큰 중에서 확률에 기반해 무작위로 하나를 선택하는 함수.
    """
    # 모델이 예측한 모든 토큰의 확률을 계산하고, 그 확률에 따라 가중치를 둔 무작위 뽑기를 1번 실행합니다.
    top_ids = weighted_scores.softmax(dim=0).multinomial(1, replacement=True)
    return top_ids

def fade_in_out(fade_in_mel, fade_out_mel, window):
    device = fade_in_mel.device
    fade_in_mel, fade_out_mel = fade_in_mel.cpu(), fade_out_mel.cpu()
    mel_overlap_len = int(window.shape[0] / 2)
    if fade_in_mel.device == torch.device('cpu'):
        fade_in_mel = fade_in_mel.clone()
    fade_in_mel[..., :mel_overlap_len] = fade_in_mel[..., :mel_overlap_len] * window[:mel_overlap_len] + \
        fade_out_mel[..., -mel_overlap_len:] * window[mel_overlap_len:]
    return fade_in_mel.to(device)


def set_all_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mask_to_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    assert mask.dtype == torch.bool
    assert dtype in [torch.float32, torch.bfloat16, torch.float16]
    mask = mask.to(dtype)
    # attention mask bias
    # NOTE(Mddct): torch.finfo jit issues
    #     chunk_masks = (1.0 - chunk_masks) * torch.finfo(dtype).min
    mask = (1.0 - mask) * -1.0e+10
    return mask


class TrtContextWrapper:
    def __init__(self, trt_engine, trt_concurrent=1, device='cuda:0'):
        self.trt_context_pool = queue.Queue(maxsize=trt_concurrent)
        self.trt_engine = trt_engine
        for _ in range(trt_concurrent):
            trt_context = trt_engine.create_execution_context()
            trt_stream = torch.cuda.stream(torch.cuda.Stream(device))
            assert trt_context is not None, 'failed to create trt context, maybe not enough CUDA memory, try reduce current trt concurrent {}'.format(trt_concurrent)
            self.trt_context_pool.put([trt_context, trt_stream])
        assert self.trt_context_pool.empty() is False, 'no avaialbe estimator context'

    def acquire_estimator(self):
        return self.trt_context_pool.get(), self.trt_engine

    def release_estimator(self, context, stream):
        self.trt_context_pool.put([context, stream])
