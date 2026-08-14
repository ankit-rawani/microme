# MicroMe 🧠

A **125-million-parameter language model built entirely from scratch on a single RTX 4060 laptop GPU (8 GB)** — no pretrained weights, no framework, ~600 lines of hand-written PyTorch. A learning project that goes end to end: tokenizer → transformer → optimizer → pretraining → chat fine-tuning → retrieval → evaluation → shipping.

🤗 **Model:** [huggingface.co/Ankitgdes/microme-125m](https://huggingface.co/Ankitgdes/microme-125m)

> **Honest scope:** at 125M params this model writes fluent English and follows chat format, but it is **not a reliable knowledge store**. The interesting part isn't the model — it's watching *how* each stage does the job the others can't.

## Results

0-shot on the base model via [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), vs standard peers in the 125M class:

| Benchmark | **MicroMe** | GPT-2 (124M) | Pythia-160m | chance |
|---|---|---|---|---|
| ARC-Easy (acc) | **52.2** | ~43.5 | ~44 | 25 |
| HellaSwag (acc_norm) | 31.9 | ~31.1 | ~30.5 | 25 |
| PIQA (acc) | 62.2 | ~62.9 | ~62 | 50 |
| LAMBADA-OpenAI (acc) | 25.5 | ~32.6 | ~33 | 0 |

Competitive with GPT-2-124M / Pythia-160m on 3 of 4 tasks **despite training on only ~3B tokens (10–100× less)**. ARC-Easy is notably strong — a direct payoff of pretraining on educational FineWeb-Edu text.

## What's inside

| File | What it is |
|---|---|
| `model.py` | single-file transformer: RMSNorm · RoPE · QK-norm · ReLU² MLP · untied embeddings · logit softcap |
| `muon.py` | Muon optimizer (Newton–Schulz orthogonalization), ~60 LOC |
| `train.py` | pretraining loop — AdamW/Muon, bf16 + `torch.compile`, resumable |
| `sft.py` | chat fine-tuning with assistant-only masked loss |
| `rag.py` · `data/facts.md` | retrieval layer + fact store (relevance-gated) |
| `serve.py` | FastAPI web chat UI with a live RAG toggle |
| `eval/lm_eval_runner.py` | lm-evaluation-harness adapter for the custom architecture |
| `tokenizer/train_bpe.py` | byte-level BPE (24,576 vocab) trained from scratch |
| `data/prepare_*.py` | stream + tokenize TinyStories / FineWeb-Edu / SmolTalk / persona |
| `hf_export.py` | package + push to the HuggingFace Hub |
| `REPORT.md` | full run log — every gate, ablation, and sample |

Data shards (`*.bin`) and checkpoints (`*.pt`) are gitignored — regenerate them with `data/prepare_*.py`.

## Quickstart

```bash
uv venv && source .venv/bin/activate
uv pip install torch numpy datasets tokenizers safetensors sentence-transformers fastapi uvicorn lm-eval

# 1. tokenizer + data  → 2. pretrain  → 3. chat SFT
python tokenizer/train_bpe.py fineweb
python data/prepare_fineweb_edu.py
python train.py run --preset micro_125m --optim muon
python data/prepare_smoltalk.py && python sft.py

# chat with it (RAG toggle in the UI) → http://localhost:8000
python serve.py

# benchmark the base model
python eval/lm_eval_runner.py
```

## The journey

`M0` smoke test → `M1` tokenizer → `M2` 30M on TinyStories → `M3` Muon beats AdamW (~1.3×) → `M4` 125M on 3B FineWeb-Edu tokens (val 2.97) → `M5` chat SFT → the **knowledge ceiling** (three fine-tunes proved you can't fine-tune facts into 125M) → **RAG** (the honest fix) → benchmarks → HuggingFace. Full story in [`REPORT.md`](REPORT.md).

**Key lesson:** style lives in the weights, facts live in retrieval. Pretraining gives language, SFT gives chat format, RAG gives facts — each doing what the others can't.

## Limitations

Not a knowledge source: confidently wrong on facts without retrieval, no arithmetic, no multi-step reasoning, no strict formats. For factual use, pair it with the RAG layer. It's for learning, experimentation, and RAG demos.

## Acknowledgements

Inspired by [nanoGPT / nanochat](https://github.com/karpathy/nanochat) (Karpathy) and the [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) speedrun community (Muon, QK-norm, ReLU²). Trained on FineWeb-Edu (ODC-By) and SmolTalk (Apache-2.0).

## License

Apache-2.0.
