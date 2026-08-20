"""Benchmark MicroMe with EleutherAI lm-evaluation-harness (plan §8).

Custom architecture, so we wrap it in an lm-eval LM adapter that scores
continuations (loglikelihood) with our GPT. Runs the standard tiny-model subset
and prints accuracies to compare against GPT-2-124M / Pythia-160m.

  python eval/lm_eval_runner.py --ckpt runs/micro_125m_muon/ckpt.pt
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse
import sys

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from lm_eval.api.model import LM
from lm_eval import simple_evaluate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import GPT, PRESETS


class MicroMeLM(LM):
    def __init__(self, ckpt, tokenizer, preset="micro_125m", device="cuda", batch_size=32):
        super().__init__()
        self._rank, self._world_size = 0, 1
        self.dev, self.batch_size = device, batch_size
        self.tok = Tokenizer.from_file(tokenizer)
        self.bos = self.tok.token_to_id("<|bos|>")
        self.model = GPT(PRESETS[preset]).to(device).eval()
        self.model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
        self.ctxlen = self.model.cfg.ctx

    def _encode_pair(self, context, continuation):
        # move trailing space of context onto continuation (HFLM convention), then
        # split the joint encoding at the context length so BPE boundaries are honored
        n = len(context) - len(context.rstrip())
        if n:
            continuation, context = context[-n:] + continuation, context[:-n]
        whole = self.tok.encode(context + continuation).ids
        ctx = self.tok.encode(context).ids if context else []
        return whole[len(ctx):]                      # continuation token ids

    @torch.no_grad()
    def loglikelihood(self, requests, disable_tqdm=False):
        enc = []
        for r in requests:
            context, continuation = r.args
            whole = self.tok.encode((context + continuation)).ids
            cont = self._encode_pair(context, continuation)
            enc.append((whole, cont))
        out = [None] * len(enc)
        order = sorted(range(len(enc)), key=lambda i: len(enc[i][0]))
        for b in range(0, len(order), self.batch_size):
            idxs = order[b:b + self.batch_size]
            inps, meta = [], []
            for i in idxs:
                whole, cont = enc[i]
                seq = ([self.bos] + whole)[-(self.ctxlen + 1):]   # bos conditions the first token
                inps.append(seq[:-1]); meta.append((i, len(cont)))
            maxlen = max(len(x) for x in inps)
            t = torch.zeros(len(inps), maxlen, dtype=torch.long, device=self.dev)
            for j, x in enumerate(inps):
                t[j, :len(x)] = torch.tensor(x, device=self.dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = self.model(t)
            lp = F.log_softmax(logits.float(), dim=-1)
            for j, (i, cl) in enumerate(meta):
                L = len(inps[j])
                sl = lp[j, L - cl:L, :]                            # cl logits predicting the continuation
                kt = torch.tensor(enc[i][1], device=self.dev)
                out[i] = (float(sl[torch.arange(cl, device=self.dev), kt].sum()),
                          bool((sl.argmax(-1) == kt).all()))
        return out

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        raise NotImplementedError("not needed for the chosen tasks")

    def generate_until(self, requests, disable_tqdm=False):
        raise NotImplementedError("not needed for the chosen tasks")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/micro_125m_muon/ckpt.pt")
    ap.add_argument("--tokenizer", default="tokenizer/fineweb-bpe.json")
    ap.add_argument("--preset", default="micro_125m")
    ap.add_argument("--tasks", default="arc_easy,hellaswag,piqa,lambada_openai")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    lm = MicroMeLM(a.ckpt, a.tokenizer, preset=a.preset)
    res = simple_evaluate(model=lm, tasks=a.tasks.split(","), limit=a.limit)
    print("\n===== RESULTS =====")
    for task, m in res["results"].items():
        acc = m.get("acc,none"); accn = m.get("acc_norm,none")
        line = f"{task:16s} acc={acc*100:.1f}%" if acc is not None else f"{task:16s} {m}"
        if accn is not None:
            line += f"  acc_norm={accn*100:.1f}%"
        print(line)
