---
license: apache-2.0
language:
- en
pipeline_tag: text-generation
tags:
- text-generation
- gpt
- trained-from-scratch
- tiny-llm
- muon
model-index:
- name: MicroMe-125M
  results:
  - task:
      type: text-generation
    dataset:
      name: ARC-Easy
      type: ai2_arc
    metrics:
    - type: acc
      value: 52.2
      name: accuracy (0-shot)
    source:
      name: lm-evaluation-harness (self-reported)
      url: https://github.com/EleutherAI/lm-evaluation-harness
  - task:
      type: text-generation
    dataset:
      name: HellaSwag
      type: hellaswag
    metrics:
    - type: acc_norm
      value: 31.9
      name: normalized accuracy (0-shot)
    source:
      name: lm-evaluation-harness (self-reported)
      url: https://github.com/EleutherAI/lm-evaluation-harness
  - task:
      type: text-generation
    dataset:
      name: PIQA
      type: piqa
    metrics:
    - type: acc
      value: 62.2
      name: accuracy (0-shot)
    source:
      name: lm-evaluation-harness (self-reported)
      url: https://github.com/EleutherAI/lm-evaluation-harness
  - task:
      type: text-generation
    dataset:
      name: LAMBADA (OpenAI)
      type: lambada_openai
    metrics:
    - type: acc
      value: 25.5
      name: accuracy (0-shot)
    source:
      name: lm-evaluation-harness (self-reported)
      url: https://github.com/EleutherAI/lm-evaluation-harness
---

# MicroMe-125M

A **125-million-parameter** GPT-style language model **trained entirely from scratch on a single RTX 4060 laptop GPU (8 GB)** — no pretrained weights, no framework, ~600 lines of hand-written PyTorch. Built as a learning project to understand how a language model is actually made, end to end: tokenizer → transformer → optimizer → pretraining → chat fine-tuning → retrieval.

> **Honest scope:** at 125M parameters this model has fluent language and follows chat format, but it is **not a reliable knowledge store** — it will state facts confidently and get them wrong. For factual use, pair it with retrieval (RAG). See *Limitations*.

## Highlights

| | |
|---|---|
| Parameters | 129.8M |
| Vocab | 24,576 (byte-level BPE, trained from scratch) |
| Context | 1024 |
| Pretraining | 3.0B tokens of FineWeb-Edu (~24× Chinchilla) |
| Base val loss | 2.97 |
| Chat val loss | 1.47 |
| Optimizer | Muon (2-D weights) + AdamW (embeddings/norms) |
| Hardware | 1× RTX 4060 Laptop, 8 GB, bf16 + torch.compile |

## Architecture

Modern-Llama skeleton with speedrun upgrades, all hand-written in `model.py`:
RMSNorm (pre-norm) · RoPE · **QK-norm** · **ReLU² MLP** (no gate) · **untied** input/output embeddings · **logit softcap** · SDPA/FlashAttention · zero-init residual projections.

## Training

1. **Tokenizer** — 24,576 byte-level BPE on FineWeb-Edu (4.44 chars/token, exact round-trip).
2. **Pretraining** — 3.0B FineWeb-Edu tokens, Muon+AdamW, warmup + linear cooldown → base val 2.97.
3. **Chat SFT** — SmolTalk (391M tokens), assistant-only masked loss → follows multi-turn conversations.
4. **Persona** — a small identity/small-talk set (with a SmolTalk anchor to avoid forgetting), so it answers greetings and knows its name is MicroMe.

The [Muon optimizer](https://kellerjordan.github.io/posts/muon/) was reimplemented by hand (~60 LOC) and beat AdamW ~1.3× on sample efficiency in a controlled ablation.

## Evaluation

0-shot on the base model via [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (self-reported), against standard peers in the 125M class. Peer numbers are approximate published values; MicroMe was run through the same harness.

| Benchmark | MicroMe-125M | GPT-2 (124M) | Pythia-160m | chance |
|---|---|---|---|---|
| ARC-Easy (acc) | **52.2** | ~43.5 | ~44 | 25 |
| HellaSwag (acc_norm) | 31.9 | ~31.1 | ~30.5 | 25 |
| PIQA (acc) | 62.2 | ~62.9 | ~62 | 50 |
| LAMBADA-OpenAI (acc) | 25.5 | ~32.6 | ~33 | 0 |

Competitive with GPT-2-124M and Pythia-160m on 3 of 4 tasks **despite training on only ~3B tokens** (10–100× less than those models). ARC-Easy is notably strong — a direct payoff of pretraining on educational FineWeb-Edu text. LAMBADA is the weak spot: last-word prediction is penalized by the small 24k vocab and the short training run.

## Intended use & limitations

**Good at:** fluent English, chat format, multi-turn context, short factual answers *when grounded with retrieval*.

**Not good at:** reliable world knowledge, arithmetic, multi-step reasoning, or strict formats (e.g. haiku). Three separate fine-tuning experiments confirmed that **you cannot fine-tune knowledge into 125M parameters** — style lives in the weights, facts must come from retrieval. Use it for learning, experimentation, and RAG demos, not as a knowledge source.

## How to use

This is a custom architecture, so load it with the bundled `model.py`:

```python
import torch, importlib.util
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer
from safetensors.torch import load_file

d = snapshot_download("Ankitgdes/microme-125m")
spec = importlib.util.spec_from_file_location("mm", f"{d}/model.py")
mm = importlib.util.module_from_spec(spec); spec.loader.exec_module(mm)

model = mm.GPT(mm.PRESETS["micro_125m"]).eval()
model.load_state_dict(load_file(f"{d}/model.safetensors"), strict=True)
tok = Tokenizer.from_file(f"{d}/tokenizer.json")
S = {k: tok.token_to_id(f"<|{k}|>") for k in ("bos","user","assistant","end","system")}

# Chat format: <|bos|><|system|>..<|end|><|user|>..<|end|><|assistant|>
# The system prompt carries MicroMe's persona — include it for identity/chitchat.
SYSTEM = "You are MicroMe, a small, friendly AI assistant. Answer briefly and helpfully."

def chat(user, max_new=60):
    ids = ([S["bos"], S["system"]] + tok.encode(SYSTEM).ids + [S["end"]]
           + [S["user"]] + tok.encode(user).ids + [S["end"], S["assistant"]])
    out = model.generate(torch.tensor([ids]), max_new, temperature=0.4, top_k=40, rep_penalty=1.3)[0].tolist()[len(ids):]
    return tok.decode(out[:out.index(S["end"])] if S["end"] in out else out)

print(chat("What is your name?"))          # -> My name is MicroMe.
print(chat("What is the largest planet?")) # weak on facts alone — pair with retrieval (RAG) for accuracy
```

For accurate factual answers, retrieve relevant context and prepend it to the user turn — a 125M model is not a reliable knowledge store on its own (see *Limitations*).

## Training data

Pretraining on [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (ODC-By, attribution) and chat/persona tuning on [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) (Apache-2.0). No share-alike data is used, so the model is released under Apache-2.0.

## Acknowledgements

Recipe inspired by nanoGPT / nanochat (Karpathy) and the modded-nanogpt speedrun community (Muon, QK-norm, ReLU²). Built as an end-to-end learning project.
