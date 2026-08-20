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
model-index:
- name: MicroMe-125M-Base
  results:
  - task: {type: text-generation}
    dataset: {name: ARC-Easy, type: ai2_arc}
    metrics:
    - {type: acc, value: 52.2, name: accuracy (0-shot)}
    source: {name: lm-evaluation-harness (self-reported), url: https://github.com/EleutherAI/lm-evaluation-harness}
  - task: {type: text-generation}
    dataset: {name: HellaSwag, type: hellaswag}
    metrics:
    - {type: acc_norm, value: 31.9, name: normalized accuracy (0-shot)}
    source: {name: lm-evaluation-harness (self-reported), url: https://github.com/EleutherAI/lm-evaluation-harness}
  - task: {type: text-generation}
    dataset: {name: PIQA, type: piqa}
    metrics:
    - {type: acc, value: 62.2, name: accuracy (0-shot)}
    source: {name: lm-evaluation-harness (self-reported), url: https://github.com/EleutherAI/lm-evaluation-harness}
  - task: {type: text-generation}
    dataset: {name: LAMBADA (OpenAI), type: lambada_openai}
    metrics:
    - {type: acc, value: 25.5, name: accuracy (0-shot)}
    source: {name: lm-evaluation-harness (self-reported), url: https://github.com/EleutherAI/lm-evaluation-harness}
---

# MicroMe-125M-Base

The pretrained **base** language model behind [**MicroMe-125M**](https://huggingface.co/Ankitgdes/microme-125m) — 125M parameters, **3B FineWeb-Edu tokens**, trained from scratch on a single RTX 4060 laptop (8 GB) in ~40 hours with hand-written PyTorch and the Muon optimizer.

This is the **raw language model** (not chat-tuned). For the chat + RAG assistant, use [Ankitgdes/microme-125m](https://huggingface.co/Ankitgdes/microme-125m). Full project + training code: [github.com/ankit-rawani/microme](https://github.com/ankit-rawani/microme).

## Benchmarks (0-shot, lm-evaluation-harness)

| ARC-Easy | HellaSwag | PIQA | LAMBADA |
|---|---|---|---|
| **52.2** | 31.9 | 62.2 | 25.5 |

Competitive with GPT-2-124M / Pythia-160m on ~10–100× less training data — ARC-Easy notably strong from FineWeb-Edu's educational text.

## Files

| File | Use |
|---|---|
| `model.safetensors` | bf16 weights — load with the bundled `model.py`, config `PRESETS["micro_125m"]` |
| `base_training_state.pt` | full checkpoint (weights + Muon/AdamW state + iter) — **resume pretraining** |
| `sft_training_state.pt` | the SmolTalk chat-SFT checkpoint (intermediate, resumable) |

## Load (base model, next-token generation)

```python
import torch, importlib.util
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

d = hf_hub_download("Ankitgdes/microme-125m-base", "model.py").rsplit("/", 1)[0]
spec = importlib.util.spec_from_file_location("mm", f"{d}/model.py")
mm = importlib.util.module_from_spec(spec); spec.loader.exec_module(mm)
model = mm.GPT(mm.PRESETS["micro_125m"]).eval()
model.load_state_dict(load_file(hf_hub_download("Ankitgdes/microme-125m-base", "model.safetensors")))
```

## License

Apache-2.0. Trained on FineWeb-Edu (ODC-By).
