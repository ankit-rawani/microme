# MicroMe — Run Report Card

- **M0 smoke** (2026-08-11 23:25) | micro_1m 0.92M params | cuda | 400 steps in 3.0s | final loss 0.0000 | gate<0.1: PASS
- **M2 micro_30m** (2026-08-12 01:42) | 41.3M | 8000 iters, 524M tok | 123.2 min | val_loss 1.3440
  - sample: "Once upon a time, there was a little girl named Lily. She loved to play outside and explore the world around her. One day, she found a small hole in the ground and decided to explore it.\n\nAs she walked through the hole, she heard a voice calling her name. It was her mom calling her for dinner. Lily was sad because she couldn't find her way back home. She decided to go back to the hole and ask her mom to call her for dinner.\n"
- **micro_30m [adamw seed0]** (2026-08-12 02:30) | 41.3M | 3000 iters, 197M tok | 44.8 min | val_loss 1.4908
  - sample: "Once upon a time, there was a little boy named Timmy. Timmy was very hungry and he wanted to eat a yummy meal. But his mommy said he needed to eat his vegetables first.\n\nTimmy didn't like vegetables because they were bitter and made him feel sick. He asked his mommy for some vegetables. But she said no and promised to make him his favorite food.\n\nLater that day, Timmy's friends came over to play. They were having a great time, but then they"
- **micro_30m [muon seed0]** (2026-08-12 03:16) | 41.3M | 3000 iters, 197M tok | 45.7 min | val_loss 1.4376
  - sample: "Once upon a time, there was a little girl named Lily. She was very curious and loved to explore. One day, she was very bored and decided to do something new.\n\nLily went outside and saw a big puddle. She wanted to jump in it and splash water everywhere. But her mom told her to be careful because it was wet.\n\nLily didn't listen and jumped in the puddle anyway. She got all wet and cold. She went back inside and told her mom what"

### Ablation rung 1 verdict (§9) — AdamW → Muon
Matched 197M-token budget (3000 iters, seed 0). Muon wins at every loss target:

| reach val loss | AdamW | Muon | Muon speedup |
|---|---|---|---|
| 2.00 | 643 it | 472 it | **1.36×** |
| 1.75 | 1171 it | 831 it | **1.41×** |
| 1.60 | 2105 it | 1622 it | **1.30×** |
| 1.50 | 2926 it | 2378 it | **1.23×** |

Final val: AdamW 1.4908, **Muon 1.4376** (−0.053). Wall-clock 44.8 vs 45.7 min (Newton-Schulz adds ~2%/step, more than repaid). Overlay: `runs/rung1_adamw_vs_muon.png`. **Confirms the classic ~1.3× Muon sample-efficiency win.** Caveat: single seed — §9 calls for 3 seeds before this is significance-grade.
- **micro_125m [muon seed1337]** (2026-08-14 06:40) | 129.8M | 5700 iters, 2988M tok | 2372.0 min | val_loss 2.9687
  - sample: 'Eating disorders can be extremely severe. When it comes to a person with an eating disorder, there are different symptoms and medical tests that can be done to assist in determining the extent of any symptoms:\n- Blood tests\n- Nutritional assessment to determine the severity of the symptoms\n- Mental status assessment to determine whether the patient is at risk of developing an eating disorder\n- An examination of the person’s teeth, teeth and jaws\n- A complete medical history (including medical history,'
- **M5 SFT micro_125m** (2026-08-14 14:38) | 5970 iters | val 1.469
  - Q: What is the capital of France?
    A[stop:False]: 'The capital of France is Paris. It is one of the largest and most vibrant cities in the world, attracting visitors from far and wide. Paris is located in the Western European region, known for its uni'
  - Q: Explain photosynthesis in one sentence.
    A[stop:False]: 'Photosynthesis is the process by which plants, algae, and other organisms convert light energy into chemical energy, known as carbon dioxide or water vapor. This process occurs in the atmosphere, as t'
  - Q: Write a haiku about the ocean.
    A[stop:False]: 'The ocean is a vibrant tapestry of color and energy, a reminder of the countless creatures that call this world home. For centuries, the inhabitants of this world have lived, worked, and died beneath '
- **M5 SFT micro_125m** (2026-08-14 16:18) | 166 iters | val 2.634
  - Q: What is the capital of France?
    A[stop:True]: 'The capital of France is Paris'
  - Q: Explain photosynthesis in one sentence.
    A[stop:True]: 'Photosynthesis is a process where plant cells absorb energy from sunlight and use it to grow using glucose as their primary energy source.'
  - Q: Write a haiku about the ocean.
    A[stop:False]: '"The Ocean is a ocean that is made of solid, and it is said to be home to alligators, rays, and krill. The ocean is also home to some of the best beaches in the world – the Maldives, Japan, and the Ga'
- **M5 SFT micro_125m** (2026-08-14 16:32) | 83 iters | val 2.644
  - Q: What is the capital of France?
    A[stop:True]: 'The capital of France is Paris.'
  - Q: Explain photosynthesis in one sentence.
    A[stop:True]: 'Photosynthesis is a process where plant cells absorb light energy from the sun, using it to convert light energy from the sun into chemical energy. This process occurs through the process of light abs'
  - Q: Write a haiku about the ocean.
    A[stop:True]: '"The Ocean is a treasure trove of all things."'

### SFT-data ablation — conciseness/knowledge (post-M5)
Same vibe-eval, seed 0, on three chat models built from the micro_125m base:
| model | recipe | result |
|---|---|---|
| micro_125m_sft | SmolTalk "all", from base | fluent, rambly; Shakespeare ✓, planet ✗ |
| micro_125m_dolly | +Dolly 15k, 4ep, lr 1e-4 | **degraded**: repetition loops, facts regressed |
| micro_125m_mix | +Dolly∪SmolTalk 50/50, 1ep, lr 5e-5 | coherence recovered; planet→Jupiter ✓, but lateral overall |
**Lesson:** SFT reshuffles *which* facts land; it cannot add reliable knowledge to 125M. Narrow terse sets (Dolly-only) cause repetition/forgetting; a mixed set + gentle LR avoids that. Real knowledge fix = RAG, not fine-tuning (§7.4).

### RAG — the knowledge fix (post-SFT-ablation)
Dense retrieval (MiniLM) over a 30-fact store + micro_125m_mix, grounded prompt. Same eval Qs, no-RAG vs RAG:
| Q | no-RAG | RAG |
|---|---|---|
| PySpark? | "a spark that creates a spark spawn" ✗ | "Apache Spark, distributed processing across a cluster" ✓ |
| FastAPI? | "fast-paced framework" ✗ | "Python web framework for building APIs, Starlette+Pydantic" ✓ |
| RAG? | "reversing an unordered set" ✗ | "retrieves documents and adds them to the prompt" ✓ |
| Romeo & Juliet? | "Peter and Juliet" ✗ | "a play by William Shakespeare" ✓ |
**Result:** RAG fixed the domain hallucinations SFT could not. Retriever reliably finds the right fact; the 125M model uses it well for definitional Qs, imperfectly when it must synthesize (occasional conflation / context-ignoring). Confirms §7.4: facts→retrieval, language→weights. Files: rag.py, data/facts.md.

**RAG polish:** similarity-threshold retrieval (drop distractors) + repetition penalty on generated tokens only (kill loops, keep faithful copying) + lower temp/tighter prompt. Result: all 7 eval Qs now answered correctly, concisely, grounded. Fixed conflation (Paris+Mumbai), loops (Shakespeare), context-ignoring (Japan). NO-RAG still hallucinates ("Pope Julius Caesar wrote Romeo and Juliet") — clean demonstration that retrieval, not weights, supplies the facts.
- **M5 SFT micro_125m** (2026-08-14 21:58) | 75 iters | val 2.607
  - Q: What is the capital of France?
    A[stop:True]: 'The capital of France is Paris.'
  - Q: Explain photosynthesis in one sentence.
    A[stop:True]: 'Photosynthesis is a process where plant cells absorb light and convert it into chemical energy.'
  - Q: Write a haiku about the ocean.
    A[stop:False]: 'Sea, a sea of blue\nThis sea is abelian\nThe sun is bright\nIt is a blue-green\nThe sea is calm\nThe ocean is peaceful\n\nIn the ocean lies a sea\nWhere all things blend\n\nWhere all things have interminated\n\n\n'
- **M5 SFT micro_125m** (2026-08-14 22:40) | 75 iters | val 1.400
  - Q: What is the capital of France?
    A[stop:True]: 'The capital of France is Paris.'
  - Q: Explain photosynthesis in one sentence.
    A[stop:False]: 'Photosynthesis is a process where sunlight, water, and other organic molecules are captured and used by plants to convert sunlight into food, oxygen, and waste products. This process is crucial for th'
  - Q: Write a haiku about the ocean.
    A[stop:True]: 'The ocean is a haven for the imagination, a haven for the imagination, and a haven for the imagination itself.'
- **M5 SFT micro_125m** (2026-08-14 22:48) | 84 iters | val 1.444
  - Q: What is the capital of France?
    A[stop:True]: 'The capital of France is Paris.'
  - Q: Explain photosynthesis in one sentence.
    A[stop:True]: 'Photosynthesis is a process where sunlight, water, and other organic molecules are captured and used by plants to convert sunlight into food, oxygen, and waste products. This process is crucial for th'
  - Q: Write a haiku about the ocean.
    A[stop:False]: "The ocean is a haven for the imagination, as it offers endless possibilities for storytelling. I've spent years studying the ocean, so I've come to realize that it's a must-have destination. The sound"

### Benchmarks — lm-evaluation-harness (base model micro_125m, 3B tokens)
| task | MicroMe | GPT-2 124M | Pythia-160m | chance |
|---|---|---|---|---|
| ARC-Easy (acc) | **52.2%** | ~43.5% | ~44% | 25% |
| HellaSwag (acc_norm) | 31.9% | ~31.1% | ~30.5% | 25% |
| PIQA (acc) | 62.2% | ~62.9% | ~62% | 50% |
| LAMBADA-OpenAI (acc) | 25.5% | ~32.6% | ~33% | 0% |
Competitive with GPT-2-124M / Pythia-160m despite ~10-100x less training. Strong on ARC-Easy (FineWeb-Edu's educational data pays off); weak on LAMBADA (short training + 24k custom vocab tokenizes target words unfavorably). Adapter: eval/lm_eval_runner.py.
