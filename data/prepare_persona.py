"""Identity + grounded-QA dataset (permissively licensed, no Dolly).

Two self-authored components, mixed with a SmolTalk (Apache-2.0) anchor:
  1. Persona/small-talk  — stable identity ("MicroMe") + greetings.
  2. Grounded QA         — teaches the model to answer FROM retrieved context
                           (the RAG skill Dolly used to provide), built from our
                           own facts.md so the whole model stays Apache-2.0-clean.
Output: data/identity/ (train + reused SmolTalk val).
"""
import random
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).parent.parent
TOK = Tokenizer.from_file(str(ROOT / "tokenizer" / "fineweb-bpe.json"))
S = {k: TOK.token_to_id(f"<|{k}|>") for k in ("bos", "user", "assistant", "end", "system")}
SYSTEM = "You are MicroMe, a small, friendly AI assistant. Answer briefly and helpfully."
FACTS = [ln.strip() for ln in open(ROOT / "data" / "facts.md") if ln.strip()]
OUT = ROOT / "data" / "identity"
PERSONA_REPEAT, GROUNDED_REPEAT, MIX_TOKENS = 40, 22, 600_000

INTENTS = {
    "name": {"q": ["What is your name?", "What's your name?", "Do you have a name?", "What are you called?",
                   "Who am I talking to?", "Tell me your name.", "May I know your name?"],
             "a": ["I'm MicroMe, a small AI assistant.", "My name is MicroMe.",
                   "I'm called MicroMe — a little AI assistant.", "MicroMe! That's my name."]},
    "who": {"q": ["Who are you?", "What are you?", "Tell me about yourself.", "Introduce yourself.",
                  "What kind of assistant are you?"],
            "a": ["I'm MicroMe, a 125-million-parameter language model trained from scratch on a laptop. I can chat and answer questions.",
                  "I'm MicroMe, a small AI assistant built from scratch as a learning project.",
                  "I'm MicroMe — a tiny language model. I can chat and answer simple questions."]},
    "greet": {"q": ["Hi", "Hello", "Hey", "Hi there", "Good morning", "Hello!", "Hey there", "Yo"],
              "a": ["Hello! I'm MicroMe. How can I help you today?", "Hi there! How can I help?",
                    "Hello! What can I do for you?", "Hey! How can I help you today?"]},
    "how": {"q": ["How are you?", "How's it going?", "How are you doing?", "How do you feel today?"],
            "a": ["I'm doing well, thanks for asking! How can I help?", "I'm good, thanks! What can I do for you?",
                  "Doing great! How can I help you today?"]},
    "can": {"q": ["What can you do?", "What are you capable of?", "How can you help me?", "What do you do?"],
            "a": ["I can chat and answer questions. I'm small, so I look things up to stay accurate.",
                  "I can have a conversation and answer simple questions.",
                  "I chat and help with simple questions. I'm a small model, so I keep answers short."]},
    "made": {"q": ["Who made you?", "Who created you?", "Who built you?", "Where do you come from?"],
             "a": ["I was built from scratch as a learning project, trained on a single laptop GPU.",
                   "I'm a small model trained from scratch as a learning project.",
                   "I was made from scratch to learn how language models work."]},
    "thanks": {"q": ["Thanks!", "Thank you", "Thanks a lot", "Thank you so much"],
               "a": ["You're welcome!", "Happy to help!", "Anytime!", "You're welcome, glad I could help."]},
    "bye": {"q": ["Bye", "Goodbye", "See you", "Bye bye"],
            "a": ["Goodbye! Take care.", "See you! Have a great day.", "Bye! Come back anytime."]},
}

# (question, answer, keyword to locate the source fact in facts.md)
GROUNDED = [
    ("What is the largest planet?", "Jupiter is the largest planet in the Solar System.", "Jupiter is the largest"),
    ("What is the second largest planet?", "Saturn is the second-largest planet.", "Saturn is the second"),
    ("Who wrote Romeo and Juliet?", "Romeo and Juliet was written by William Shakespeare.", "Shakespeare"),
    ("What is the capital of France?", "The capital of France is Paris.", "Paris is the capital"),
    ("What is FastAPI?", "FastAPI is a modern Python web framework for building APIs.", "FastAPI is a modern"),
    ("What is PySpark used for?", "PySpark is the Python API for Apache Spark, used for distributed processing of large datasets.", "PySpark is the Python API"),
    ("What is retrieval-augmented generation?", "RAG is a technique where a model retrieves relevant documents and adds them to its prompt to ground its answers.", "Retrieval-augmented generation (RAG)"),
    ("Why is the sky blue?", "The sky looks blue because air molecules scatter blue light more than other colours (Rayleigh scattering).", "sky appears blue"),
    ("What is the capital of Japan?", "The capital of Japan is Tokyo.", "capital of Japan is Tokyo"),
    ("When is the best time to visit Japan?", "The best times to visit Japan are spring for cherry blossoms and autumn for mild weather.", "best times to visit Japan"),
    ("What is the capital of India?", "New Delhi is the capital of India.", "New Delhi is the capital"),
    ("What is the largest ocean?", "The Pacific Ocean is the largest and deepest ocean.", "Pacific Ocean is the largest"),
    ("What is the highest mountain?", "Mount Everest is the highest mountain above sea level.", "Mount Everest"),
    ("At what temperature does water boil?", "Water boils at 100 degrees Celsius at sea level.", "Water boils at 100"),
    ("What is the speed of light?", "The speed of light in a vacuum is about 299,792 kilometres per second.", "speed of light"),
    ("How many bones are in the human body?", "The adult human body has 206 bones.", "206 bones"),
    ("What is Docker?", "Docker packages software into containers so it runs the same way anywhere.", "Docker is a platform"),
    ("What is Kubernetes?", "Kubernetes automates the deployment and scaling of containerized applications.", "Kubernetes is an open-source"),
    ("What is Git?", "Git is a distributed version-control system that tracks changes in source code.", "Git is a distributed"),
    ("What is SQL?", "SQL is the standard language for querying relational databases.", "SQL (Structured Query"),
    ("What is machine learning?", "Machine learning is a branch of AI where systems learn patterns from data.", "Machine learning is a branch"),
    ("What is a transformer in AI?", "A transformer is a neural network architecture based on self-attention.", "A transformer is a neural"),
    ("Who developed the theory of relativity?", "Albert Einstein developed the theory of relativity.", "Albert Einstein"),
    ("What is the capital of the United States?", "The capital of the United States is Washington, D.C.", "Washington, D.C"),
    ("What is photosynthesis?", "Photosynthesis is how green plants use sunlight to turn carbon dioxide and water into glucose and oxygen.", "Photosynthesis is the process"),
    ("What causes rain?", "Rain forms when water vapour cools, condenses into droplets, and falls when heavy enough.", "Rain forms when"),
    ("What is the Taj Mahal?", "The Taj Mahal is a white marble mausoleum in Agra, India.", "Taj Mahal"),
    ("What is the largest river?", "The Amazon River is the largest river by water discharge.", "Amazon River"),
    ("What is DNA?", "DNA carries the genetic instructions for living organisms.", "DNA (deoxyribonucleic"),
    ("What is Apache Spark?", "Apache Spark is an engine for processing big data in parallel across a cluster.", "Apache Spark processes"),
]


def encode(ids, mask, tokens, learn):
    e = TOK.encode(tokens).ids
    ids += e; mask += [1 if learn else 0] * len(e)


def render(user, asst):
    ids, mask = [S["bos"]], [0]
    sc = TOK.encode(SYSTEM).ids;  ids += [S["system"]] + sc + [S["end"]];    mask += [0] * (len(sc) + 2)
    uc = TOK.encode(user).ids;    ids += [S["user"]] + uc + [S["end"]];      mask += [0] * (len(uc) + 2)
    ac = TOK.encode(asst).ids;    ids += [S["assistant"]] + ac + [S["end"]]; mask += [0] + [1] * (len(ac) + 1)
    return ids, mask


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    pool = []
    persona = [(q, a) for it in INTENTS.values() for q in it["q"] for a in it["a"]]
    pool += persona * PERSONA_REPEAT

    # grounded QA: user turn = "answer from context", context = source fact + 2 distractors (shuffled)
    for _ in range(GROUNDED_REPEAT):
        for q, a, key in GROUNDED:
            src = next(f for f in FACTS if key in f)
            ctx = [src] + rng.sample([f for f in FACTS if f != src], 2)
            rng.shuffle(ctx)
            user = ("Answer the question using only the context. Be concise and factual.\n\nContext:\n"
                    + "\n".join(f"- {c}" for c in ctx) + f"\n\nQuestion: {q}")
            pool.append((user, a))
    rng.shuffle(pool)

    ids, mask = [], []
    for u, a in pool:
        i, m = render(u, a); ids += i; mask += m
    ids, mask = np.array(ids, dtype=np.uint16), np.array(mask, dtype=np.uint8)

    mb = np.asarray(np.memmap(ROOT / "data" / "smoltalk" / "train.bin", dtype=np.uint16, mode="r")[:MIX_TOKENS])
    mm = np.asarray(np.memmap(ROOT / "data" / "smoltalk" / "train.mask", dtype=np.uint8, mode="r")[:MIX_TOKENS])
    np.concatenate([ids, mb]).tofile(OUT / "train.bin")
    np.concatenate([mask, mm]).tofile(OUT / "train.mask")
    for s in ("val.bin", "val.mask"):
        (OUT / s).write_bytes((ROOT / "data" / "smoltalk" / s).read_bytes())
    print(f"identity train: {len(ids)+len(mb):,} tokens | persona {len(persona)}x{PERSONA_REPEAT} + "
          f"grounded {len(GROUNDED)}x{GROUNDED_REPEAT} + smoltalk {len(mb):,}")


if __name__ == "__main__":
    main()
