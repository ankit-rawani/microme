"""Byte-level BPE tokenizer (plan §3). One tokenizer per corpus.

  python tokenizer/train_bpe.py tinystories   # M1/M2 shakedown corpus
  python tokenizer/train_bpe.py fineweb        # M4 main run (FineWeb-Edu)

vocab 24,576, byte-level (full 256-byte alphabet => exact round-trip),
special tokens reserved up front. Trained on ~2 GB of the target corpus.
"""
import sys
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

VOCAB = 24576
SPECIALS = ["<|bos|>", "<|user|>", "<|assistant|>", "<|end|>", "<|system|>"]
TRAIN_BYTES = 2_000_000_000  # ~2 GB (§3)

CORPORA = {
    "tinystories": dict(repo="roneneldan/TinyStories", config=None,
                        split="train", eval_split="validation", field="text", out="tinystories-bpe.json"),
    "fineweb":     dict(repo="HuggingFaceFW/fineweb-edu", config="sample-100BT",
                        split="train", eval_split="train", field="text", out="fineweb-bpe.json"),
}


def stream(spec, split):
    return load_dataset(spec["repo"], name=spec["config"], split=split, streaming=True)


def train_iter(spec):
    seen = 0
    for ex in stream(spec, spec["split"]):
        t = ex[spec["field"]]
        seen += len(t)
        if seen > TRAIN_BYTES:
            break
        yield t


def train(spec):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB, special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=True)
    print(f"training BPE (vocab {VOCAB}) on ~{TRAIN_BYTES/1e9:.0f}GB of {spec['repo']}...")
    tok.train_from_iterator(train_iter(spec), trainer)
    out = Path(__file__).parent / spec["out"]
    tok.save(str(out))
    print("saved", out, "| vocab", tok.get_vocab_size())
    return tok


def accept(tok, spec):
    docs, seen = [], 0
    for ex in stream(spec, spec["eval_split"]):   # in-sample is fine for a compression/round-trip gate
        docs.append(ex[spec["field"]])
        if len(docs) >= 10_000:
            break
    bad = sum(1 for d in docs if tok.decode(tok.encode(d).ids) != d)
    ratio = sum(len(d) for d in docs) / sum(len(tok.encode(d).ids) for d in docs)
    print(f"round-trip mismatches: {bad}/{len(docs)}")
    print(f"compression: {ratio:.3f} chars/token (gate >= 3.7)")
    for sid in SPECIALS:
        assert tok.token_to_id(sid) is not None, f"missing special {sid}"
    assert bad == 0 and ratio >= 3.7, f"acceptance failed (bad={bad}, ratio={ratio:.3f})"
    print("acceptance: PASS")
    return ratio


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "tinystories"
    spec = CORPORA[name]
    accept(train(spec), spec)
