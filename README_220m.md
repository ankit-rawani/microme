---
license: apache-2.0
language:
- en
pipeline_tag: text-generation
tags:
- text-generation
- gpt
- trained-from-scratch
- base-model
- muon
- tied-embeddings
model-index:
- name: MicroMe-220M
  results:
  - task: {type: text-generation}
    dataset: {name: ARC-Easy, type: ai2_arc}
    metrics:
    - {type: acc, value: 53.6, name: accuracy (0-shot)}
    source: {name: lm-evaluation-harness (self-reported), url: https://github.com/EleutherAI/lm-evaluation-harness}
  - task: {type: text-generation}
    dataset: {name: HellaSwag, type: hellaswag}
    metrics:
    - {type: acc_norm, value: 33.0, name: normalized accuracy (0-shot)}
    source: {name: lm-evaluation-harness (self-reported), url: https://github.com/EleutherAI/lm-evaluation-harness}
  - task: {type: text-generation}
    dataset: {name: PIQA, type: piqa}
    metrics:
    - {type: acc, value: 62.5, name: accuracy (0-shot)}
    source: {name: lm-evaluation-harness (self-reported), url: https://github.com/EleutherAI/lm-evaluation-harness}
  - task: {type: text-generation}
    dataset: {name: LAMBADA (OpenAI), type: lambada_openai}
    metrics:
    - {type: acc, value: 26.3, name: accuracy (0-shot)}
    source: {name: lm-evaluation-harness (self-reported), url: https://github.com/EleutherAI/lm-evaluation-harness}
---

# MicroMe-220M

A **217M-parameter** deep-thin GPT trained from scratch on a single RTX 4060 laptop (8 GB), the scaled-up capstone of the [MicroMe](https://github.com/ankit-rawani/microme) project. It applies the recipe that a controlled 3-seed ablation picked out at small scale: **tied input/output embeddings** (a parameter-efficiency win at this vocab-to-size ratio) and the **Muon optimizer**, with **gradient checkpointing** to fit 217M on 8 GB.

Trained on **2B FineWeb-Edu tokens** in ~38 hours. It reaches a lower validation loss (2.896) than the [125M base](https://huggingface.co/Ankitgdes/microme-125m-base) (2.969) on **one-third less data**, and improves on it across all four benchmarks.

## Benchmarks (0-shot, lm-evaluation-harness)

| | 217M | 125M | GPT-2 (124M) | Pythia-160m |
|---|---|---|---|---|
| ARC-Easy | **53.6** | 52.2 | ~43.5 | ~44 |
| HellaSwag (norm) | **33.0** | 31.9 | ~31.1 | ~30.5 |
| PIQA | 62.5 | 62.2 | ~62.9 | ~62 |
| LAMBADA | 26.3 | 25.5 | ~32.6 | ~33 |

Beats GPT-2-124M / Pythia-160m on ARC-Easy and HellaSwag, on far less training data. LAMBADA stays the weak spot (small 24k custom vocab, short training run). This is a **base** model, not chat-tuned; it is not a reliable knowledge store on its own (pair with retrieval for facts).

## Architecture
28 layers, d=768, 12 heads, ctx 1024, vocab 24,576. RMSNorm, RoPE, QK-norm, ReLU2 MLP, **tied embeddings**, logit softcap. Details and training code: [github.com/ankit-rawani/microme](https://github.com/ankit-rawani/microme).

## Files
| File | Use |
|---|---|
| `model.safetensors` | bf16 weights (load with the bundled `model.py`, `PRESETS["mini_220m"]`) |
| `training_state.pt` | full checkpoint (weights + Muon/AdamW state + iter) to resume pretraining |

## Load
```python
import torch, importlib.util
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

d = hf_hub_download("Ankitgdes/microme-220m", "model.py").rsplit("/", 1)[0]
spec = importlib.util.spec_from_file_location("mm", f"{d}/model.py")
mm = importlib.util.module_from_spec(spec); spec.loader.exec_module(mm)
model = mm.GPT(mm.PRESETS["mini_220m"]).eval()
model.load_state_dict(load_file(hf_hub_download("Ankitgdes/microme-220m", "model.safetensors")))
```

## License
Apache-2.0. Trained on FineWeb-Edu (ODC-By).
