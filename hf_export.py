"""Export a checkpoint into a HuggingFace-ready model repo, and optionally push.

  python hf_export.py --ckpt runs/micro_125m_persona/ckpt.pt --out hf         # build bundle
  python hf_export.py --out hf --push <user>/microme-125m --private           # upload existing bundle

Custom architecture (not a standard HF class), so the repo ships model.py and a
load snippet in the card. Weights are saved bf16 to keep the download small.
"""
import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import save_file

from model import GPT, PRESETS

ROOT = Path(__file__).parent


def build(ckpt, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    cfg = PRESETS["micro_125m"]
    model = GPT(cfg)
    sd = torch.load(ckpt, map_location="cpu")["model"]
    model.load_state_dict(sd)
    # weights as bf16 safetensors
    save_file({k: v.to(torch.bfloat16).contiguous() for k, v in model.state_dict().items()},
              str(out / "model.safetensors"), metadata={"format": "pt"})
    (out / "config.json").write_text(json.dumps(
        {"model_type": "microme", "architecture": "GPT (custom)", **asdict(cfg),
         "note": "Load with the bundled model.py; see README."}, indent=2))
    shutil.copy(ROOT / "model.py", out / "model.py")
    shutil.copy(ROOT / "tokenizer" / "fineweb-bpe.json", out / "tokenizer.json")
    shutil.copy(ROOT / "README_hf.md", out / "README.md")
    mb = (out / "model.safetensors").stat().st_size / 1e6
    print(f"built {out}/  (weights {mb:.0f} MB bf16) — files: {[p.name for p in out.iterdir()]}")


def push(out, repo_id, private):
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_folder(folder_path=str(out), repo_id=repo_id, commit_message="Add MicroMe-125M")
    print(f"pushed -> https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--out", default="hf")
    ap.add_argument("--push")            # repo id, e.g. user/microme-125m
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()
    if a.ckpt:
        build(a.ckpt, a.out)
    if a.push:
        push(a.out, a.push, a.private)
