# AdaHiP Attention

[HiP Attention](https://github.com/DeepAuto-AI/hip-attention)의 희소 검색 예산 `k`를 attention entropy에 따라 조절하는 학부 연구 프로토타입입니다. 고정 예산 대신 입력의 attention 분포를 이용해 검색량을 정하는 방법을 실험합니다.

## 예산 결정 방식

최근 query·key의 causal attention 분포에서 정규화 entropy를 계산하고, 한 번의 layer forward에 적용할 scalar 예산을 정합니다.

```text
x = clamp(alpha * mean_normalized_entropy, 0, 1) ** power
k = RoundUpTo64(k_min + (k_max - k_min) * x)
k_min = max(64, base_k / 4), k_max = 2 * base_k
```

HiP의 계층적 token pruning과 Triton kernel을 기반으로 entropy probe, 검색 예산 연결, 모델 연동, GPU/Docker 실험·집계 파이프라인을 구현했습니다. 기본 runner는 하단 3개 layer에서 dense attention을 사용하는 hybrid 구성입니다.

## 코드 구조

| 경로 | 내용 |
|---|---|
| `src/hip_attn/utils/entropy_probe.py` | causal prefix로 제한한 entropy와 GQA head 처리 |
| `src/hip_attn/utils/attention.py` | scalar 예산을 HiP 단계별 검색량에 연결 |
| `src/hip_attn/models/modeling_llama.py` | AdaHiP의 native GQA 및 RoPE 경로 |
| `hip-research/src/hip_research/main/jobs/ppl.py` | 유효 next-token 수로 가중한 NLL/PPL 평가 |
| `run_experiments.sh` | method·context·dataset·alpha/power 실험 |
| `aggregate_summaries.py` | metric summary를 CSV로 집계 |

## 환경과 실행

Linux, Python 3.10–3.12, CUDA와 호환되는 PyTorch/Triton/FlashAttention 환경을 사용합니다. 모델 접근 권한과 데이터 다운로드가 필요합니다. 의존성은 `pyproject.toml`·`uv.lock`, 컨테이너 실행은 Docker 스크립트를 참고하세요.

```bash
git clone https://github.com/hyeongus2/adahip-attention.git
cd adahip-attention
uv sync
uv run python -m hip_research.main.model_eval --help
METHODS="fa2 hip adahip" DATASETS_PPL="pg19" STRIDES="32768" KS="512" COUNT=2 bash run_experiments.sh
python aggregate_summaries.py --in_dir results/summaries --out_dir results/aggregate
```

runner 기본값은 Llama-3.1-8B-Instruct, NF4 4-bit 양자화, dense layer 3개, `HIP_EXTEND=0`입니다. 비교 실험에서는 method 외의 모델·양자화·dense layer·표본·환경 조건을 동일하게 맞춥니다.

## 평가 방식과 실험 상태

PPL은 `exp(total_nll / valid_tokens)`로 계산합니다. 마지막 짧은 구간도 실제 target 수로 가중하고, 여러 sample의 loss는 평균합니다. 집계기는 이 평가 방식의 `token_weighted_v1` 메타데이터가 있는 summary를 사용합니다.

CPU 합성 테스트는 미래 key 차단, prefix 정규화, GQA head 등가성, decode·짧은 probe, token 가중 PPL과 집계를 다룹니다.

```bash
python -m unittest discover -s tests -p "test_adahip_cpu.py" -v
```

현재 구현의 제약은 다음과 같습니다.

- 예산은 query별 값이 아닌 layer 전체의 scalar입니다. prefill에서는 여러 query의 entropy를 평균하므로 앞선 query의 예산이 뒤 query에 영향을 받을 수 있습니다.
- entropy probe는 padding 없는 연속 causal sequence를 전제로 합니다.
- CUDA sparse kernel 정확도, native GQA GPU 통합, 장문 PPL·latency·메모리 비교는 추가 실험이 필요합니다.

현재 성능 비교 수치는 제공하지 않습니다. 모델 품질과 실행 비용 평가는 예산 선택의 인과성, kernel 정확도를 확인한 뒤 모델·데이터 revision과 원시 로그를 함께 기록하는 실험으로 진행합니다.

## 출처와 라이선스

기반 연구는 [HiP Attention](https://github.com/DeepAuto-AI/hip-attention)입니다. 사용·재배포 조건은 [LICENSE.md](LICENSE.md)를 참고하세요. 기반 코드의 라이선스와 저작자 고지가 적용됩니다.
