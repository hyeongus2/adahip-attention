#!/usr/bin/env python3
"""
env_report.py
Collect a reproducible "Section 5: Experimental Evaluation" environment snapshot.

Run:
  python env_report.py

Optional:
  HF_MODEL_ID=meta-llama/Meta-Llama-3.1-8B-Instruct python env_report.py
  HF_REVISION=<commit_or_tag> HF_MODEL_ID=... python env_report.py
  TEST_TOKENIZE_ONLY=1 python env_report.py
"""

import os
import sys
import json
import platform
import subprocess
from datetime import datetime

def run_cmd(cmd: list[str], timeout_sec: int = 20) -> dict:
    """Run a shell command and return stdout/stderr/returncode."""
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        return {
            "cmd": " ".join(cmd),
            "returncode": p.returncode,
            "stdout": p.stdout.strip(),
            "stderr": p.stderr.strip(),
        }
    except FileNotFoundError:
        return {
            "cmd": " ".join(cmd),
            "returncode": None,
            "stdout": "",
            "stderr": "COMMAND_NOT_FOUND",
        }
    except subprocess.TimeoutExpired:
        return {
            "cmd": " ".join(cmd),
            "returncode": None,
            "stdout": "",
            "stderr": f"TIMEOUT>{timeout_sec}s",
        }

def print_header(title: str):
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)

def safe_import(name: str):
    try:
        mod = __import__(name)
        return mod, None
    except Exception as e:
        return None, repr(e)

def main():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print_header("SECTION 5 ENVIRONMENT REPORT (AdaHiP / HiP / FA2)")
    print(f"Timestamp: {ts}")
    print(f"Python executable: {sys.executable}")
    print(f"Python version    : {sys.version.replace(os.linesep, ' ')}")
    print(f"Platform          : {platform.platform()}")
    print(f"Machine/Arch      : {platform.machine()} / {platform.processor()}")
    print(f"Working dir       : {os.getcwd()}")
    print(f"ENV: VIRTUAL_ENV  : {os.getenv('VIRTUAL_ENV')}")
    print(f"ENV: CONDA_PREFIX : {os.getenv('CONDA_PREFIX')}")
    print(f"ENV: CUDA_VISIBLE_DEVICES : {os.getenv('CUDA_VISIBLE_DEVICES')}")

    # ------------------------------------------
    # OS / Kernel
    # ------------------------------------------
    print_header("OS / KERNEL")
    print(json.dumps(run_cmd(["uname", "-a"]), indent=2, ensure_ascii=False))
    if os.path.exists("/etc/os-release"):
        try:
            with open("/etc/os-release", "r", encoding="utf-8") as f:
                print("\n/etc/os-release:\n" + f.read().strip())
        except Exception as e:
            print(f"Failed to read /etc/os-release: {e!r}")

    # ------------------------------------------
    # CPU / Memory (best-effort)
    # ------------------------------------------
    print_header("CPU / MEMORY (best-effort)")
    print(json.dumps(run_cmd(["bash", "-lc", "lscpu | sed -n '1,25p'"]), indent=2, ensure_ascii=False))
    print(json.dumps(run_cmd(["bash", "-lc", "free -h"]), indent=2, ensure_ascii=False))

    # ------------------------------------------
    # GPU / Driver / CUDA runtime
    # ------------------------------------------
    print_header("GPU / DRIVER / CUDA")
    print(json.dumps(run_cmd(["bash", "-lc", "nvidia-smi"]), indent=2, ensure_ascii=False))
    print(json.dumps(run_cmd(["bash", "-lc", "nvidia-smi -L"]), indent=2, ensure_ascii=False))
    print(json.dumps(run_cmd(["bash", "-lc", "nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv"]), indent=2, ensure_ascii=False))
    print(json.dumps(run_cmd(["bash", "-lc", "nvcc --version"]), indent=2, ensure_ascii=False))

    # ------------------------------------------
    # PyTorch / CUDA / cuDNN / bf16
    # ------------------------------------------
    print_header("PYTORCH / CUDA / cuDNN")
    torch, torch_err = safe_import("torch")
    if torch is None:
        print(f"torch import failed: {torch_err}")
    else:
        info = {
            "torch.__version__": torch.__version__,
            "torch.version.cuda": getattr(torch.version, "cuda", None),
            "torch.cuda.is_available": torch.cuda.is_available(),
            "torch.backends.cudnn.version": torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None,
            "torch.backends.cuda.matmul.allow_tf32": getattr(torch.backends.cuda.matmul, "allow_tf32", None),
            "torch.backends.cudnn.allow_tf32": getattr(torch.backends.cudnn, "allow_tf32", None) if hasattr(torch.backends, "cudnn") else None,
        }
        if torch.cuda.is_available():
            try:
                info.update({
                    "gpu_name": torch.cuda.get_device_name(0),
                    "gpu_capability": torch.cuda.get_device_capability(0),
                    "bf16_supported": torch.cuda.is_bf16_supported(),
                    "device_count": torch.cuda.device_count(),
                })
            except Exception as e:
                info["gpu_query_error"] = repr(e)
        print(json.dumps(info, indent=2, ensure_ascii=False))

        # PyTorch SDP / Flash kernel availability (best-effort)
        sdp = {}
        try:
            # Newer torch versions
            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "sdp_kernel"):
                # Torch 2.1+ has context manager; availability differs
                sdp["has_torch.backends.cuda.sdp_kernel"] = True
            else:
                sdp["has_torch.backends.cuda.sdp_kernel"] = False
        except Exception as e:
            sdp["sdp_check_error"] = repr(e)
        print("\nSDP/Flash Kernel (best-effort):")
        print(json.dumps(sdp, indent=2, ensure_ascii=False))

    # ------------------------------------------
    # Key Python packages (versions)
    # ------------------------------------------
    print_header("KEY PYTHON PACKAGES (VERSIONS)")
    packages = [
        "transformers", "tokenizers", "accelerate", "datasets", "safetensors",
        "flash_attn", "triton", "numpy", "scipy",
    ]
    versions = {}
    for name in packages:
        mod, err = safe_import(name)
        if mod is None:
            versions[name] = {"available": False, "error": err}
        else:
            ver = getattr(mod, "__version__", None)
            versions[name] = {"available": True, "version": ver}
    print(json.dumps(versions, indent=2, ensure_ascii=False))

    # pip/uv snapshot
    print_header("PACKAGE SNAPSHOT (pip/uv)")
    print(json.dumps(run_cmd(["bash", "-lc", "pip --version"]), indent=2, ensure_ascii=False))
    print(json.dumps(run_cmd(["bash", "-lc", "uv --version"]), indent=2, ensure_ascii=False))
    print(json.dumps(run_cmd(["bash", "-lc", "pip freeze | sed -n '1,120p'"]), indent=2, ensure_ascii=False))

    # ------------------------------------------
    # Optional: model load sanity (Transformers)
    # ------------------------------------------
    print_header("OPTIONAL: MODEL LOAD SANITY (Transformers)")
    model_id = os.getenv("HF_MODEL_ID", "").strip()
    revision = os.getenv("HF_REVISION", "").strip() or None
    tokenize_only = os.getenv("TEST_TOKENIZE_ONLY", "0") == "1"

    if model_id == "":
        print("Skipped: HF_MODEL_ID is not set.")
        print("To run model sanity, set e.g.:")
        print("  HF_MODEL_ID=meta-llama/Meta-Llama-3.1-8B-Instruct python env_report.py")
    else:
        transformers, terr = safe_import("transformers")
        if transformers is None:
            print(f"transformers import failed: {terr}")
        else:
            from transformers import AutoTokenizer
            print(f"HF_MODEL_ID : {model_id}")
            print(f"HF_REVISION : {revision}")
            try:
                tok = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
                sample = "Hello! This is a quick sanity check for Section 5 environment reporting."
                enc = tok(sample, return_tensors="pt")
                print("Tokenizer loaded OK.")
                print(f"Tokenized sample length: {enc['input_ids'].shape[-1]}")
                if tokenize_only:
                    print("TEST_TOKENIZE_ONLY=1 -> stop after tokenizer.")
                else:
                    # Try model load (heavy). Only do if CUDA exists; else CPU.
                    from transformers import AutoModelForCausalLM
                    import torch as _torch

                    dtype = _torch.float16 if (_torch.cuda.is_available()) else _torch.float32
                    device_map = "auto" if _torch.cuda.is_available() else None
                    print(f"Loading model (this can be heavy). dtype={dtype}, device_map={device_map}")

                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        revision=revision,
                        torch_dtype=dtype,
                        device_map=device_map,
                        trust_remote_code=True,
                    )
                    model.eval()

                    # Minimal forward pass (no generation)
                    with _torch.no_grad():
                        if _torch.cuda.is_available():
                            enc = {k: v.to(model.device) for k, v in enc.items()}
                        out = model(**enc)
                    print("Model forward OK.")
                    # Print config highlights useful for paper
                    cfg = model.config.to_dict() if hasattr(model, "config") else {}
                    keys = ["model_type", "architectures", "hidden_size", "num_hidden_layers", "num_attention_heads", "max_position_embeddings"]
                    paper_cfg = {k: cfg.get(k, None) for k in keys}
                    print("Model config highlights:")
                    print(json.dumps(paper_cfg, indent=2, ensure_ascii=False))

            except Exception as e:
                print(f"Model sanity failed: {e!r}")

    print_header("DONE")
    print("Copy-paste this entire output into chat for Section 5 write-up refinement.")

if __name__ == "__main__":
    main()
