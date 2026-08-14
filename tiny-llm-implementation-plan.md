# Project "MicroMe" — Train a Tiny Personal LLM on an RTX 4060 Laptop

**Audience:** coding agents (Claude Code or similar) executing this plan end-to-end, plus the human owner (Ankit) reading along to learn.
**Prime directive:** the goal is *learning how an LLM is made*, so every stage must be readable, hackable, and instrumented — never a black-box library call when a ~200-line implementation teaches more.

---

## 0. Hardware & Environment Constraints (non-negotiable)

| Resource | Budget | Consequence |
|---|---|---|
| GPU | NVIDIA RTX 4060 Laptop, **8 GB VRAM** | From-scratch pretraining capped at ~50–160M params; QLoRA fine-tuning possible up to ~3–7B |
| System RAM | **16 GB** | Never load a corpus into RAM. All token data stored as memory-mapped `uint16` binary shards (nanoGPT style) |
| OS | Windows laptop (HP Omen) | **Use WSL2 + Ubuntu 24.04** for all training. `torch.compile`, Triton, and most CUDA tooling are far more reliable on Linux. Keep data on the WSL ext4 filesystem, not `/mnt/c` (9p I/O is slow) |
| Realistic sustained compute | ~10–15 TFLOPS effective bf16 | Budget wall-clock with `FLOPs ≈ 6 × params × tokens` |

Environment setup (Stage 0 acceptance = all of these pass):

```bash
# inside WSL2 Ubuntu
uv venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install numpy datasets tiktoken sentencepiece wandb transformers trl peft bitsandbytes
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())"
```

Notes:
- Use PyTorch SDPA (`F.scaled_dot_product_attention`) — it dispatches to FlashAttention kernels automatically. Do **not** fight to compile `flash-attn` from source.
- bf16 autocast everywhere; fp32 master weights in the optimizer.
- Log every run to Weights & Biases (or a local CSV fallback): loss curves are the primary learning artifact.

---

## 1. Strategy Overview — Two Tracks, One Repo

**Track A — Build & pretrain from scratch (the learning core).** Implement tokenizer → transformer → training loop → inference engine by hand, closely following the *nanochat* pipeline (Karpathy's 2025 successor to nanoGPT: a minimal full-stack ChatGPT clone covering tokenization, pretraining, midtraining, SFT, optional RL, evaluation, and serving). We reimplement the ideas at 4060 scale rather than cloning blindly.

**Track B — Knowledge distillation (the "extract knowledge from a big model" part).** Two teacher options:

1. **Frontier API teacher (Claude/GPT) — black-box distillation.** You can only get *text*, not logits. So distillation = synthetic-data SFT (sequence-level KD): teacher writes high-quality instruction/response pairs, student imitates.
   - ⚠️ **Terms-of-service note:** OpenAI's and Anthropic's consumer/API terms restrict using outputs to develop competing models. A private, non-commercial learning project is the lowest-risk end of the spectrum, but the owner should read the current terms and decide. This document defaults to option 2 for the heavy lifting.
2. **Local open-weights teacher (recommended default): Qwen3-4B-Instruct or Llama-3.2-3B-Instruct, 4-bit quantized (~2.5–3 GB VRAM).** This unlocks *white-box* distillation — real logits, reverse-KL, and on-policy methods — which is both licence-clean (check each model's licence; Qwen is Apache-2.0) and teaches far more about how distillation actually works.

**The "groundbreaking" angle.** A hobby project won't beat frontier labs, but it *can* do genuine mini-science by adopting the modded-nanogpt speedrun mindset: fixed target loss, fixed hardware, then A/B every idea. The speedrun community drove GPT-2-class training from 45 min → ~2–3 min on 8×H100 with a stack of techniques (Muon optimizer, RoPE, QK-norm, ReLU², untied embeddings, value/embedding skip connections, logit softcap, sliding-window schedules), and Muon has since been adopted for frontier-scale runs like Kimi K2 and GLM-4.5. We import the highest-value, lowest-complexity items from that stack (§4) and run our own ablation ladder (§9).

---

## 2. Repository Layout

```
microme/
├── configs/              # one .yaml per run; every run is reproducible from its config
├── data/
│   ├── prepare_tinystories.py
│   ├── prepare_fineweb_edu.py     # streams, tokenizes, writes .bin shards
│   ├── gen_distill_data.py        # teacher → synthetic SFT jsonl
│   └── prepare_persona.py         # owner's own writing → style corpus
├── tokenizer/
│   └── train_bpe.py               # byte-level BPE, HF tokenizers lib
├── model.py                       # single-file transformer (~350 LOC)
├── muon.py                        # Muon optimizer (~60 LOC, Newton–Schulz)
├── train.py                       # pretraining loop
├── sft.py                         # chat-format supervised fine-tuning
├── gkd.py                         # on-policy distillation vs local teacher (TRL GKDTrainer or hand-rolled)
├── sample.py                      # CLI chat with KV cache
├── eval/
│   ├── val_loss.py
│   └── lm_eval_runner.py          # lm-evaluation-harness subset
├── export_gguf.py                 # llama.cpp conversion + Q4 quant
└── REPORT.md                      # auto-appended run report card (nanochat-style)
```

Rules for the coding agent:
- `model.py` must stay a single readable file. No abstraction layers, no framework.
- Every stage appends metrics to `REPORT.md` (config hash, tokens seen, wall-clock, val loss, sample generations). This gamified report is the owner's learning log.
- Checkpoint every 30 min of wall-clock; training must be resumable (laptop = interruptions).

---

## 3. Tokenizer (Stage 1)

- Byte-level BPE trained on ~2 GB of the pretraining corpus.
- **Vocab size 24,576** (not 50k/65k): at ≤125M params, a large vocab wastes a huge parameter fraction on the embedding matrix. 24k keeps embeddings ≤20% of params while compressing English + code decently.
- Special tokens reserved up front: `<|bos|>`, `<|user|>`, `<|assistant|>`, `<|end|>`, `<|system|>`.
- Acceptance: round-trip encode/decode identity on 10k random docs; compression ratio ≥3.7 chars/token on held-out English.

## 4. Model Architecture (Stage 2)

Modern-Llama skeleton with the cheap, proven speedrun upgrades:

| Choice | Setting | Why |
|---|---|---|
| Norm | RMSNorm, pre-norm | standard, stable |
| Positional | RoPE | standard |
| Attention | MHA with **QK-norm**; SDPA kernel | QK-norm = stability at high LR, ~3 lines |
| MLP | **ReLU²** (`relu(x)**2`), no gate | speedrun-preferred; simpler + faster than SwiGLU at this scale |
| Embeddings | **Untied** input/output embeddings | consistent speedrun win; costs params, gains loss |
| Head | Logit softcap (tanh, cap 15–30) | stability |
| Init | scaled residual init, zero-init proj layers | speedrun practice |
| Shape | **Deep-and-thin** | MobileLLM finding: at sub-1B scale, depth beats width |

Two model configs:

- **`micro-30m`** (TinyStories run): 8 layers, d=448, 7 heads, ctx 512. ≈30M params.
- **`micro-125m`** (main run): 20 layers, d=640, 10 heads, ctx 1024. ≈125M params, GPT-2-small class.

VRAM sanity for `micro-125m`: bf16 weights 0.25 GB + fp32 optimizer states ~1.0–1.5 GB + grads + activations → comfortably <6 GB at micro-batch 8–16 × grad-accum to effective batch ~0.5M tokens.

Stretch items (only after baseline works): value-embedding skip connections, sliding-window/full attention interleave, FP8 head. Each goes through the ablation ladder (§9), never straight into the baseline.

## 5. Optimizer (Stage 3) — Muon + AdamW hybrid

- **Muon** (momentum + Newton–Schulz orthogonalization) on all 2-D hidden weight matrices; **AdamW** on embeddings, unembedding, norms, scalars. This is the exact split used by the speedrun and by Kimi K2-scale runs; Muon's claim to fame is winning every nanoGPT record since Oct 2024 on sample efficiency.
- Implement `muon.py` by hand (~60 LOC) — this is one of the highest learning-per-line files in the repo.
- LR: Muon ~0.02 (scaled by matrix shape), AdamW ~3e-3 embeddings / 6e-4 head equivalents; warmup 300 steps; linear cooldown for final 40% of steps. Momentum warmup 0.85→0.95.
- Fallback: if Muon misbehaves, plain AdamW β=(0.9, 0.95), wd 0.1 — and that A/B itself becomes ablation #1.

## 6. Pretraining Data & Runs (Stage 4)

**Run 1 — `micro-30m` on TinyStories (~500M–1B tokens, 2–5 hrs).**
Purpose: end-to-end pipeline shakedown + the single most instructive result in tiny-LM research — a 30M model trained on a narrow high-quality distribution produces fluent, coherent English. Data quality beats scale; this is the phi/TinyStories lesson experienced firsthand.

**Run 2 — `micro-125m` on FineWeb-Edu (2.5–3B tokens, ~2–3 days wall-clock).**
- Stream `HuggingFaceFW/fineweb-edu` (sample-100BT config), tokenize on the fly, write `uint16` shards; total disk ~6 GB, RAM footprint ~constant.
- 3B tokens ≈ 24× Chinchilla-optimal for 125M — deliberate heavy overtraining, the SmolLM recipe: for a model you'll actually *use*, tokens-per-param far beyond compute-optimal keeps paying.
- Mix in ~5% code (the-stack-smol) + ~5% simple QA to help later chat behavior.
- Acceptance: val loss ≤3.4 on FineWeb-Edu held-out (GPT-2-ish ballpark for this budget); sensible 200-token continuations logged in REPORT.md.

## 7. Midtraining + Distillation (Stage 5) — where the "knowledge extraction" happens

1. **Chat-format midtraining:** ~300–500k conversations from SmolTalk (nanochat's midtrain set), rendered with our special tokens. Teaches turn structure, instruction following, "I don't know" behavior.
2. **Black-box teacher synthesis (`gen_distill_data.py`):**
   - Seed topics: owner's actual domains (RAG, PySpark, FastAPI, motorcycles, Indian daily life) + general knowledge taxonomy.
   - Teacher generates Q/A pairs and short explanations *targeted at what a 125M model can absorb*: short, factual, single-hop. Long chain-of-thought is wasted on a student this small.
   - Filter: dedupe (MinHash), length caps, teacher self-critique pass.
   - ~50–100k pairs. If using Claude/GPT as teacher, see ToS note §1; the persona/style data below is the least sensitive use.
3. **White-box on-policy distillation (`gkd.py`)** with local Qwen3-4B (4-bit) as teacher:
   - Plain SFT on teacher text = off-policy, and students then compound their own errors at inference (exposure bias). **GKD / on-policy distillation** fixes this: the *student* generates continuations, the *teacher* scores every token, and the loss is a divergence (generalized JSD; reverse-KL per MiniLLM keeps the student from smearing mass over teacher low-probability regions).
   - Practical loop on 8 GB: teacher runs 4-bit via llama.cpp server or transformers+bnb (~3 GB), student bf16 (~0.5 GB + optimizer). Alternate: sample batch from student → get teacher logprobs → distill step. TRL's `GKDTrainer` is the reference; hand-roll a minimal version first for learning.
   - **This SFT-vs-GKD comparison is the project's flagship experiment** (§9).
4. **Persona ("LLM of myself"):** collect owner's own writing (notes, messages, blog drafts) → style-transfer SFT pass (teacher rewrites generic answers in owner's voice; student trains on those). Honest expectation: a 125M model can *mimic style and tone*; it cannot be a reliable store of personal facts — Karpathy's own advice for nanochat-class models is to use RAG for personal knowledge rather than fine-tuning it in. Style lives in weights; facts live in retrieval.
5. Optional: small DPO pass on ~5k preference pairs (teacher-ranked) for politeness/formatting.

## 8. Evaluation (Stage 6)

- Primary: held-out val loss / perplexity per stage (the honest metric at this scale).
- lm-evaluation-harness subset: ARC-Easy, HellaSwag, PIQA, LAMBADA — expect barely-above-chance to low-30s%; the point is *measuring movement between ablations*, not the absolute number.
- Distillation-specific: win-rate of GKD-student vs SFT-student on 200 held-out prompts, judged by the teacher (blind, position-swapped).
- Vibe eval: fixed 25-prompt sheet (stories, simple math, owner-domain questions, persona prompts) regenerated after every stage into REPORT.md.

## 9. The Ablation Ladder (the "mini-groundbreaking" program)

Fixed protocol: `micro-30m` config, fixed 0.8B-token budget, fixed seed set (3 seeds), report mean val loss + wall-clock. One change per rung:

1. AdamW → Muon (expect the classic ~1.3× sample-efficiency win)
2. SwiGLU → ReLU²
3. Tied → untied embeddings
4. Wide-shallow (12L×d576) vs deep-thin (20L×d448) at matched params
5. FineWeb-Edu vs FineWeb (quality ablation)
6. Vocab 16k vs 24k vs 32k
7. SFT-only vs SFT+GKD (flagship)

Each rung = one REPORT.md entry with loss curve overlay. If any rung shows a surprising result at 3-seed significance, that's a legitimate blog post — this is exactly how speedrun records and optimizer research actually progress.

## 10. Inference & Ship (Stage 7)

- `sample.py`: KV-cache decode, temperature/top-p, streaming CLI chat.
- Export to GGUF via llama.cpp `convert_hf_to_gguf.py` (write a small HF-format exporter first), quantize Q4_K_M → model runs on CPU/phone-class hardware.
- Optional weekend extra: tiny FastAPI + web chat UI (owner's home turf).

## 11. Milestones & Gates (execution order for coding agents)

| # | Milestone | Gate to pass before next |
|---|---|---|
| M0 | WSL2 env + smoke train (1M-param model overfits 1 batch to ~0 loss) | loss <0.1 on memorization test |
| M1 | Tokenizer trained | §3 acceptance |
| M2 | `micro-30m` TinyStories run | coherent 100-token story samples |
| M3 | Muon integrated | ablation rung 1 complete |
| M4 | `micro-125m` pretrain | val loss ≤3.4 |
| M5 | Midtrain + SFT chat model | follows 3-turn conversations |
| M6 | GKD distillation | beats M5 on blind win-rate |
| M7 | Persona pass + DPO | style match on vibe sheet |
| M8 | GGUF export + chat UI | Q4 model chats on CPU |

**Total wall-clock estimate: ~1 week of GPU time spread over 3–4 weekends.**

## 12. What the Owner Should Learn at Each Stage (why this teaches "what goes into an LLM")

1. *Tokenization* — why vocab size is a parameter-budget decision, not a detail.
2. *Architecture* — every block hand-written; QK-norm/RoPE/RMSNorm cease to be jargon.
3. *Optimization* — Muon vs AdamW makes optimizer geometry concrete.
4. *Scaling & data* — Chinchilla vs overtraining, quality-vs-quantity, felt on your own electricity bill.
5. *Post-training* — the pretrain→midtrain→SFT→distill→DPO ladder is the same shape as every frontier lab pipeline, just 4 orders of magnitude smaller.
6. *Distillation* — off-policy vs on-policy, black-box vs white-box, and why logits matter.
7. *Serving* — KV cache and quantization explain why inference costs what it costs.
