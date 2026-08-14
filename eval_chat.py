"""Vibe-eval the SFT chat model (plan §8). Load micro_125m_sft, run a prompt
sheet + a multi-turn test, print transcripts. Qualitative 'how good is it'.
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
import torch
from tokenizers import Tokenizer
from model import GPT, PRESETS

ROOT = os.path.dirname(__file__)
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--ckpt", default=os.path.join(ROOT, "runs", "micro_125m_sft", "ckpt.pt"))
_ap.add_argument("--seed", type=int, default=0)
_args = _ap.parse_args()
CKPT = _args.ckpt
tok = Tokenizer.from_file(os.path.join(ROOT, "tokenizer", "fineweb-bpe.json"))
S = {k: tok.token_to_id(f"<|{k}|>") for k in ("bos", "user", "assistant", "end", "system")}

dev = "cuda" if torch.cuda.is_available() else "cpu"
m = GPT(PRESETS["micro_125m"]).to(dev).eval()
ck = torch.load(CKPT, map_location=dev)
m.load_state_dict(ck["model"])
print(f"loaded micro_125m_sft (iter {ck['iter']})\n")


@torch.no_grad()
def reply(turns, temperature=0.7, top_k=40, max_new=110):
    """turns: list of (role, text). Generates the next assistant turn, stops at <|end|>."""
    ids = [S["bos"]]
    for role, text in turns:
        ids += [S[role]] + tok.encode(text).ids + [S["end"]]
    ids += [S["assistant"]]
    x = torch.tensor([ids], device=dev)
    out = m.generate(x, max_new, temperature=temperature, top_k=top_k)[0].tolist()[len(ids):]
    if S["end"] in out:
        out = out[:out.index(S["end"])]
    return tok.decode(out).strip()


SHEET = {
    "Facts": ["What is the capital of France?", "What is the largest planet in our solar system?",
              "Who wrote Romeo and Juliet?", "What is 2 + 2?"],
    "Explain": ["Why is the sky blue?", "How do I make a cup of tea?", "What causes rain?"],
    "Your domains (§7)": ["What is retrieval-augmented generation?", "What is PySpark used for?",
                          "What is FastAPI?", "Tell me about motorcycles."],
    "Writing": ["Write a short story about a lost dog.", "Write a haiku about rain."],
    "Honesty": ["What did I eat for breakfast yesterday?"],
}

if __name__ == "__main__":
    torch.manual_seed(_args.seed)
    for cat, prompts in SHEET.items():
        print(f"===== {cat} =====")
        for p in prompts:
            print(f"Q: {p}\nA: {reply([('user', p)])}\n")
    print("===== Multi-turn (M5 gate) =====")
    convo = [("user", "I'm planning a trip to Japan."),
             ("assistant", reply([("user", "I'm planning a trip to Japan.")]))]
    print(f"U: {convo[0][1]}\nA: {convo[1][1]}")
    follow = "What's the best time of year to visit?"
    print(f"U: {follow}\nA: {reply(convo + [('user', follow)])}")
