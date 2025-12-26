import os

import torch
from hip_research.dataset.passkey import Passkey
from hip_research.models.sglang_model import SglangModel
from tqdm import tqdm

import warnings
try:
    from vllm import LLM, SamplingParams
except ModuleNotFoundError:
    LLM = torch.Tensor  # sentinel type for isinstance checks
    SamplingParams = None
    warnings.warn("vllm is not installed. passkey job will not support vLLM backend.")


def get_numbers(s, cnt):
    lst = [c for c in s if c.isdigit()]
    # print(lst, s)
    if len(lst) < cnt:
        lst += ["_"] * (cnt - len(lst))
    return "".join(lst)


def job_passkey(args, model, tokenizer, device):
    dataset = Passkey(tokenizer, batch_size=args.batch_size)

    accuracy = dict()
    seq_lens = set()
    locations = set()

    for j, (input_ids, target_ids) in enumerate(
        tqdm(dataset, dynamic_ncols=True, leave=False)
    ):
        if isinstance(model, LLM):
            input_ids = input_ids.cuda()
            target_ids = target_ids.cuda()

            prompts = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
            sampling_params = SamplingParams(
                n=1,
                temperature=0.0,
                top_k=1,
                max_tokens=20,
            )

            outputs = model.generate(prompts, sampling_params, use_tqdm=False)
            output = []
            for item in outputs:
                output.append(item.outputs[0].text)
        elif isinstance(model, SglangModel):
            input_text = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
            IS_CHAT = os.getenv("IS_CHAT", "0") == "1"
            if IS_CHAT:
                output = [
                    model.generate(
                        input_text=input_text[0],
                        max_tokens=1024,
                        need_chat_prompt=True,
                        system_message=None,
                        handle_deepseek=True,
                        verbose=False,
                    )
                ]
                print(output)
            else:
                output = [model.generate(input_text=input_text, max_tokens=20)]
        else:
            # input_ids = input_ids.cuda()
            # target_ids = target_ids.cuda()

            # with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            #     output = model.generate(
            #         input_ids,
            #         max_new_tokens=20,
            #         min_new_tokens=5,
            #         do_sample=False,
            #         num_beams=1,
            #         attention_mask=None,
            #         # pad_token_id=tokenizer.eos_token_id,
            #     )
            #     for m in model.modules():
            #         if hasattr(m, "_clean_cache"):
            #             m._clean_cache()
            #     output = output[:, input_ids.shape[1] :]
            #     tqdm.write(f"{tokenizer.batch_decode(output)}")


# ============================================================
            # [FINAL FIX] Align to Head Dimension (128)
            # 이유: Llama-3 Head Dim이 128이므로, view() 연산 시 
            # 전체 토큰 수가 128의 배수여야 함.
            # ============================================================
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            # 1. 원본 데이터 보존
            raw_input_ids = input_ids
            raw_len = raw_input_ids.shape[1]
            
            # [수정 1] Attention Mask 초기화 (원본 데이터 부분은 1로 설정)
            attention_mask = torch.ones_like(input_ids, device=device)
            
            # [수정 핵심] 32가 아니라 128의 배수로 맞춤
            required_alignment = max(int(args.block_size_q), 128)
            
            remainder = raw_len % required_alignment
            pad_len = 0
            
            # print(f"[DEBUG] Raw Len: {raw_len}, Alignment: {required_alignment}, Remainder: {remainder}")

            if remainder != 0:
                pad_len = required_alignment - remainder
                
                # 패딩 토큰 결정
                pad_val = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                
                # [Left Padding] for Input IDs
                pad_tensor = torch.full(
                    (raw_input_ids.shape[0], pad_len), 
                    pad_val, 
                    dtype=raw_input_ids.dtype, 
                    device=device
                )
                input_ids = torch.cat([pad_tensor, raw_input_ids], dim=1)

                # [수정 2] Attention Mask에도 Left Padding 적용 (패딩 부분은 0으로 설정)
                # 모델이 앞부분의 패딩 토큰을 무시하도록 만들기 위함
                pad_mask = torch.zeros(
                    (raw_input_ids.shape[0], pad_len),
                    dtype=attention_mask.dtype,
                    device=device
                )
                attention_mask = torch.cat([pad_mask, attention_mask], dim=1)

                print(f"[DEBUG] Padded Len: {input_ids.shape[1]} (Multiple of {required_alignment})")

            # 2. 모델 생성
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                output = model.generate(
                    input_ids,
                    attention_mask=attention_mask,  # [수정 3] 생성된 마스크 전달 (기존 None 제거)
                    max_new_tokens=len(target_ids[0]),
                    min_new_tokens=1,
                    do_sample=False,
                    num_beams=1,
                    # attention_mask=None,  # [수정] 기존 코드 주석 처리됨
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                    use_cache=True,
                )

                for m in model.modules():
                    if hasattr(m, "_clean_cache"):
                        m._clean_cache()

                # 3. 결과 보정 (패딩 제거)
                output = output[:, input_ids.shape[1] :]
                
                tqdm.write(f"{tokenizer.batch_decode(output)}")

            # 4. 원본 input_ids 복구 (정확도 계산을 위해)
            input_ids = raw_input_ids
        ################################################################################

        # tqdm(tokenizer.batch_decode(output))
        truth = tokenizer.batch_decode(target_ids)
        est = [
            get_numbers(s.strip(), 5)[:5]
            for s in (
                output if isinstance(output[0], str) else tokenizer.batch_decode(output)
            )
        ]

        t = tokenizer.batch_decode(input_ids)[0]  # type: str
        e = truth[0]  # type: str
        idx = t.find(e)

        location = idx / len(t)
        location = int(location / 0.2) / 5

        seq_len = input_ids.shape[1]

        seq_lens.add(seq_len)
        locations.add(location)

        accuracy_key = (seq_len, location)
        acc_sum, acc_count = accuracy.get(accuracy_key, (0, 0))
        for x, y in zip(truth, est):
            for cx, cy in zip(x, y):
                if cx == cy:
                    acc_sum += 1
                acc_count += 1
        accuracy[accuracy_key] = (acc_sum, acc_count)

        accuracy_key = (seq_len,)
        acc_sum, acc_count = accuracy.get(accuracy_key, (0, 0))
        for x, y in zip(truth, est):
            for cx, cy in zip(x, y):
                if cx == cy:
                    acc_sum += 1
                acc_count += 1
        accuracy[accuracy_key] = (acc_sum, acc_count)

        tqdm.write(
            f"current accuracy { {k: f'{v[0] / (v[1] + 1e-20)*100:.2f}' for k, v in accuracy.items()} } | {truth[0]}, {est[0]}"
        )

    seq_lens = list(sorted(seq_lens))
    locations = list(sorted(locations))

    def to_acc(x):
        return x[0] / (x[1] + 1e-20) * 100

    print(f'Loc.,{",".join(map(lambda x: str(x), seq_lens))}')
    print(
        f'Avg.,{",".join([str(to_acc(accuracy[(seq_len,)])) for seq_len in seq_lens])}'
    )
    for location in locations:
        print(
            f'{location},{",".join([str(to_acc(accuracy[(seq_len, location)])) for seq_len in seq_lens])}'
        )
