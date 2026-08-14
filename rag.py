"""Minimal RAG (plan §7.4 / honest knowledge fix).

Facts live in a lookup, the 125M model handles language. Retrieve top-k facts
by dense-embedding similarity, inject them into the prompt, let the chat model
answer grounded in real data instead of guessing from its weights.

  python rag.py                 # side-by-side no-RAG vs RAG on the eval questions
  python rag.py "your question"
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tokenizers import Tokenizer
from model import GPT, PRESETS

ROOT = os.path.dirname(__file__)
CKPT = os.path.join(ROOT, "runs", "micro_125m_clean", "ckpt.pt")   # identity + grounded RAG, Apache-2.0-clean lineage
dev = "cuda" if torch.cuda.is_available() else "cpu"

# knowledge base
FACTS = [ln.strip() for ln in open(os.path.join(ROOT, "data", "facts.md")) if ln.strip()]
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=dev)
FEMB = embedder.encode(FACTS, normalize_embeddings=True)          # (N, 384)

# chat model
tok = Tokenizer.from_file(os.path.join(ROOT, "tokenizer", "fineweb-bpe.json"))
S = {k: tok.token_to_id(f"<|{k}|>") for k in ("bos", "user", "assistant", "end", "system")}
SYSTEM = "You are MicroMe, a small, friendly AI assistant. Answer briefly and helpfully."
GATE = 0.40   # only ground in retrieval when the top fact clears this similarity (chitchat sits ~0.1)
m = GPT(PRESETS["micro_125m"]).to(dev).eval()
m.load_state_dict(torch.load(CKPT, map_location=dev)["model"])


def retrieve(q, k=2, thresh=0.30):
    qv = embedder.encode([q], normalize_embeddings=True)[0]
    sims = FEMB @ qv
    idx = np.argsort(sims)[::-1][:k]
    keep = [i for i in idx if sims[i] >= thresh] or [idx[0]]   # drop distractors, always keep top-1
    return [FACTS[i] for i in keep], sims[keep]


@torch.no_grad()
def gen(user, system=SYSTEM, max_new=90, temperature=0.3, top_k=40, rep_penalty=1.3):
    ids = [S["bos"]]
    if system:
        ids += [S["system"]] + tok.encode(system).ids + [S["end"]]
    ids += [S["user"]] + tok.encode(user).ids + [S["end"], S["assistant"]]
    out = m.generate(torch.tensor([ids], device=dev), max_new, temperature=temperature,
                     top_k=top_k, rep_penalty=rep_penalty)[0].tolist()[len(ids):]
    if S["end"] in out:
        out = out[:out.index(S["end"])]
    return tok.decode(out).strip()


def answer(q, k=2):
    ctx, sims = retrieve(q, k)
    if len(sims) == 0 or sims[0] < GATE:      # no relevant fact -> plain chat, don't force irrelevant grounding
        return gen(q), []
    grounded = ("Answer the question using only the context. Be concise and factual.\n\nContext:\n"
                + "\n".join(f"- {c}" for c in ctx) + f"\n\nQuestion: {q}")
    return gen(grounded), ctx


EVAL = ["What is the capital of France?", "What is the largest planet in our solar system?",
        "Who wrote Romeo and Juliet?", "What is retrieval-augmented generation?",
        "What is PySpark used for?", "What is FastAPI?", "What is the best time to visit Japan?"]

if __name__ == "__main__":
    torch.manual_seed(0)
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        a, ctx = answer(q)
        print(f"Q: {q}\nretrieved: {ctx[0][:80]}...\nA: {a}")
    else:
        for q in EVAL:
            no_rag = gen(q)
            rag, ctx = answer(q)
            print(f"===== {q} =====")
            print(f"  NO-RAG: {no_rag[:200]}")
            print(f"  RAG   : {rag[:200]}")
            print(f"  (top fact: {ctx[0][:90]}...)\n")
