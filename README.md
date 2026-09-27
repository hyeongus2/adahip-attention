# AdaHiP Attention

[HiP Attention](https://github.com/DeepAuto-AI/hip-attention)을 기반으로 attention entropy에 따라 희소 검색 예산 `k`를 조절하는 학부 연구 프로토타입입니다. 심현성이 생성형 AI의 분석·코드 작성 도움을 받아 기존 구현에 통합하고 GPU/Docker 실험 파이프라인을 구성했습니다.

## 구현 범위

최근 query·key의 causal attention 분포에서 정규화 entropy를 계산하고, 한 번의 **layer forward 전체에 적용하는 scalar 예산**을 결정합니다. query마다 다른 예산을 전달하는 kernel은 구현하지 않았습니다.

```text
x = clamp(alpha * mean_normalized_entropy, 0, 1) ** power
k = RoundUpTo64(k_min + (k_max - k_min) * x)
k_min = max(64, base_k / 4), k_max = 2 * base_k
```

HiP의 계층적 token pruning·Triton kernel·모델 연결을 상속하며, 본인 변경은 entropy probe, 예산 연결, 실험 runner·집계·표·환경 점검 도구 중심입니다. 기본 runner는 하단 3개 layer를 dense attention으로 처리하는 hybrid 구성이므로 모델 전체가 subquadratic이라고 설명하지 않습니다.

## 코드

| 경로 | 내용 |
|---|---|
| `src/hip_attn/utils/entropy_probe.py` | causal prefix로 제한한 entropy와 GQA head 처리 |
| `src/hip_attn/utils/attention.py` | scalar 예산을 HiP 단계별 검색량에 연결 |
| `src/hip_attn/models/modeling_llama.py` | AdaHiP의 native GQA 및 RoPE 경로 |
| `hip-research/src/hip_research/main/jobs/ppl.py` | 유효 next-token 수로 가중한 NLL/PPL 평가 |
| `run_experiments.sh` | method·context·dataset·alpha/power 실험 |
| `aggregate_summaries.py` | versioned metric summary를 CSV로 집계 |

## 환경과 실행

GPU 전체 실행은 Linux, Python 3.10–3.12, CUDA 및 호환되는 PyTorch/Triton/FlashAttention 환경이 필요합니다. 모델 접근 권한과 데이터 다운로드를 별도로 준비합니다. 기존 `pyproject.toml`·`uv.lock`과 Docker 스크립트를 보존합니다.

```bash
git clone https://github.com/hyeongus2/adahip-attention.git
cd adahip-attention
uv sync
uv run python -m hip_research.main.model_eval --help
METHODS="fa2 hip adahip" DATASETS_PPL="pg19" STRIDES="32768" KS="512" COUNT=2 bash run_experiments.sh
python aggregate_summaries.py --in_dir results/summaries --out_dir results/aggregate
```

runner 기본값은 Llama-3.1-8B-Instruct, NF4 4-bit 양자화, dense layer 3개, `HIP_EXTEND=0`입니다. method·model·양자화·dense layer·표본·환경을 같게 맞춰 비교해야 합니다. 위 명령은 실행 예시이며 이 정리 과정에서 GPU 재실험을 수행하지 않았습니다.

## 검증 상태

2026-09-28 수정에서는 미래 key를 보던 entropy probe, AdaHiP의 불필요한 KV head 반복, 누적 PPL을 다시 평균하던 집계를 바로잡았습니다. PPL은 `exp(total_nll / valid_tokens)`로 계산하며 마지막 짧은 구간도 실제 target 수로 가중합니다. 여러 sample은 최소 loss 대신 평균 loss를 사용합니다. 새 집계기는 `token_weighted_v1` 메타데이터가 없는 과거 summary를 재사용하지 않습니다.

CPU PyTorch 환경에서는 CUDA kernel을 불러오지 않고 다음 합성 회귀검증을 실행할 수 있습니다.

```bash
python -m unittest discover -s tests -p "test_adahip_cpu.py" -v
```

검증 범위는 causal 미래 key 차단, prefix 정규화, GQA head 등가성, decode·짧은 probe, token 가중 PPL과 집계입니다. 실제 sparse kernel 정확도, native GQA GPU 통합, 장문 PPL·latency·메모리는 재검증하지 않았습니다. probe는 padding 없는 연속 causal sequence를 전제로 합니다. 또한 한 layer의 여러 query에서 얻은 값을 하나의 예산으로 평균하므로, prefill 전체의 예산 선택이 엄밀한 autoregressive causality를 만족한다고 검증한 것은 아닙니다. 이 구조의 개선과 kernel 검증 전에는 PPL을 모델 품질의 확정 근거로 사용하지 않습니다.

**기존 README·학위논문의 성능표와 속도 우위 주장은 철회했습니다.** 원래 논문과 표는 과거 이력 자료이며 현재 검증 결과로 인용하지 않습니다. 수정 코드로 공통 조건의 원시 로그·환경·모델/데이터 revision을 보존한 재실험 후 성능을 다시 보고할 예정입니다.

## 출처와 라이선스

원 [HiP Attention](https://github.com/DeepAuto-AI/hip-attention) 연구와 저작자 고지를 유지합니다. 사용·재배포 조건은 저장소의 [LICENSE.md](LICENSE.md)를 확인하세요. 상속한 코드에는 원 라이선스와 저작자 고지가 그대로 적용됩니다.
