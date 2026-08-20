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
- **M0 smoke** (2026-08-17 00:56) | micro_1m 0.92M | cuda | 400 steps 2.8s | loss 0.0000 | gate<0.1: PASS
- **micro_30m [swigtest seed1337]** (2026-08-17 01:00) | 41.2M | 20 iters, 1M tok | 0.8 min | val_loss 8.5328
  - sample: '!" cozymied Skye Austr puffy teammate bluebergrand treetop procla scrambledMya league Zap positting hard lessongr shimmiedFoolish yeast woman Davis freed totabitha farms teleamine Hmm Isabelle tieHelpFr dents Iron Lab DoctorsSuzie slapped Help Bones Nora hustle ou�ory Frankie z picn Sudha Tony Barbie oarslaud Biggs Puppies up hudd cigarette 10ered buttercup stream cows potions invitation embarra Kelsey shining Wilburlit Janetcka menacingly GlJosh Museum Pole Jane Puppies Voice spee gem onto wise rabbitâaughtyaccoon wherever beetouble awardedBud photosurs Bingo reminded'
- **micro_30m [baseline seed0]** (2026-08-17 01:47) | 41.3M | 3000 iters, 197M tok | 46.2 min | val_loss 1.4366
  - sample: 'Once upon a time, there was a little boy named Timmy. Timmy was very hungry and wanted some food. He went to his mom and asked if he could have some bread and milk.\n\nMommy said, "Yes, Timmy. You can have some bread and milk."\n\nTimmy was happy and started to eat. Suddenly, he saw a cat in the park. The cat was very cute and Timmy wanted to pet it.\n\nHe asked the cat, "Can I pet you'
- **micro_30m [swiglu seed0]** (2026-08-17 02:35) | 41.2M | 3000 iters, 197M tok | 47.2 min | val_loss 1.4412
  - sample: 'Once upon a time, there was a little girl named Lily. She was very hungry and wanted some food. She went to her mom and asked if she could have some food. Her mom said yes and gave her a plate.\n\nLily ate all of the food and didn\'t want it. But then, she saw a man with a big bag. He said, "You can\'t leave my food here. It\'s not safe." Lily felt sad and hungry.\n\nLater that day'
- **micro_30m [reuse2 seed0]** (2026-08-17 03:50) | 41.3M | 3000 iters, 197M tok | 75.0 min | val_loss 1.4267
  - sample: 'Once upon a time, there was a little girl named Lily. She was very hungry and wanted some food. She went to her mom and asked if she could have some bread and milk. Her mom said yes and gave her a lovely piece of bread.\n\nLily then went to the park and saw a bird. She wanted to catch the bird, but it flew away. She saw a flower and thought it would be fun to catch it. She walked over to the flower, but the bird'
- **micro_30m [tied seed0]** (2026-08-17 04:36) | 30.3M | 3000 iters, 197M tok | 45.9 min | val_loss 1.4302
  - sample: 'Once upon a time, there was a little girl named Lily. She had a new toy box that she loved to play with. One day, her friend, Tim, came over to play. They decided to play with the toy box together.\n\nWhile they were playing, Lily saw a spark in the box. She picked it up and asked Tim, "What does spark mean?" Tim replied, "It makes a flame." They were excited to see the flame and started to play with it'

### Architecture ablation lab (§9) — micro_30m, TinyStories, Muon, 3000 iters, seed 0
One arch change per run vs the shipped baseline (ReLU², untied, QK-norm). Overlay: runs/ablate_arch.png.
| variant | params | final val | vs baseline | wall-clock |
|---|---|---|---|---|
| baseline (ReLU², untied) | 41.3M | 1.4366 | — | 48 min |
| SwiGLU (param-matched) | 41.2M | 1.4412 | +0.0046 (worse) | 48 min |
| layer_reuse=2 (MobileLLM) | 41.3M | 1.4267 | **−0.0099** (best) | 75 min (2× compute) |
| tied embeddings | **30.3M** | 1.4302 | **−0.0064** | 46 min |
**Findings (single seed → directional):** (1) ReLU² ≥ SwiGLU at this scale — validates the speedrun choice. (2) **Tied embeddings win: same-or-better loss at 27% fewer params** — contradicts the "untied wins" speedrun rule *because at 24k vocab / 41M params the embeddings are ~54% of params*, so tying is a big param-efficiency gain (scale/vocab-dependent). (3) Layer reuse helps marginally but costs 2× compute — a param-efficiency win, not compute-efficiency. Margins are small (≤0.01) at 3000 iters near the loss floor; 3 seeds needed for significance.
- **micro_30m [baseline seed1]** (2026-08-17 09:01) | 41.3M | 3000 iters, 197M tok | 45.7 min | val_loss 1.4310
  - sample: 'Once upon a time there was a mighty girl. She loved to play.\n\nOne day, she found a needle. It was shiny and silver. She liked it so much that she wanted to give it a hug. So she hugged it tight and it made her feel so happy.\n\nThe girl wanted to show her needle to her friends. So she decided to put it in a bag. She ran to her friends and they were so excited.\n\nThe girl put her needle in'
- **micro_30m [tied seed1]** (2026-08-17 09:46) | 30.3M | 3000 iters, 197M tok | 45.5 min | val_loss 1.4258
  - sample: 'Once upon a time there was a mighty girl. She loved to play.\n\nOne day, she found a rope. It was big and long and she liked it very much.\n\nThe girl went to the park. She loved to swing on the rope and feel the wind in her hair.\n\nThen, she heard a loud noise. She turned around and saw a big monster. It was huge and had lots of teeth.\n\nThe girl was scared. She wanted to run'
- **micro_30m [reuse2 seed1]** (2026-08-17 10:59) | 41.3M | 3000 iters, 197M tok | 72.9 min | val_loss 1.4197
  - sample: 'Once upon a time there was a mighty girl. She loved to play. Her favorite thing to do was to go on adventures. \nOne day, she was playing in the park when she heard a loud noise. It sounded like something shaking. \nThe girl was scared. She asked her mom what it was. Her mom said it was a monster. \nThe girl was curious. She wanted to find out. She asked her mom, but her mom said no. \nSo'
- **micro_30m [baseline seed2]** (2026-08-17 11:44) | 41.3M | 3000 iters, 197M tok | 45.0 min | val_loss 1.4344
  - sample: 'Once there was a boy who loved to write. He wrote stories, drew pictures, and even drew a picture of a bird. His favorite thing to do was to write and write. One day he had a special letter. It was a piece of steel. He could not believe it.\n\nHe opened the letter and saw that it said "T". He was very excited. He opened the letter and found that it said "I am going to write something special today".\n\nHe was'
- **micro_30m [tied seed2]** (2026-08-17 12:29) | 30.3M | 3000 iters, 197M tok | 44.6 min | val_loss 1.4283
  - sample: "Once upon a time, there was a little girl named Lily. She loved to play with her toys and eat her yummy food. One day, her mom brought home a big, red apple from the store. Lily didn't like it, but her mom told her it was important to eat healthy food too.\n\nLily's mom warned her that the apple was not good for her. But Lily didn't listen and took a big bite anyway. The apple was so delicious that she started to cry"
- **micro_30m [reuse2 seed2]** (2026-08-17 13:43) | 41.3M | 3000 iters, 197M tok | 74.4 min | val_loss 1.4208
  - sample: "Once upon a time, there was a little girl named Lily. She loved to play with her toys and eat her yummy food. One day, Lily's mom told her they were going to a party. Lily was so excited! She wore her favorite dress and had a big smile on her face.\n\nWhen they arrived at the party, Lily saw lots of people and decorations. She saw lots of balloons and decorations. But then, a big boy came and grabbed a toy. Lily didn't"

### Architecture ablation — 3-seed confirmation (§9)
Seeds {0,1,2}, micro_30m/TinyStories/Muon/3000 iters. Final val loss:
| variant | params | s0 | s1 | s2 | mean ± std | vs baseline |
|---|---|---|---|---|---|---|
| baseline | 41.3M | 1.4366 | 1.4310 | 1.4344 | 1.4340 ± 0.0028 | — |
| tied embeds | **30.3M** | 1.4302 | 1.4258 | 1.4283 | 1.4281 ± 0.0022 | −0.0059 (σ 0.0036) **SURVIVES** |
| layer_reuse=2 | 41.3M | 1.4267 | 1.4197 | 1.4208 | 1.4224 ± 0.0038 | −0.0116 (σ 0.0047) **SURVIVES** |
Clean per-seed separation (every tied/reuse2 seed beats every baseline seed). **Tied embeddings = significance-grade win at 27% fewer params** — contradicts the speedrun's "untied wins" *at this ≤41M / 24k-vocab regime* where embeddings are ~54% of params. Layer reuse helps too but costs 2× compute (param- not compute-efficiency). Files: ablate.sh, ablate_seeds.sh, ablate_seeds_stats.py.
- **mini_220m [muon seed1337]** (2026-08-20 02:49) | 217.1M | 3815 iters, 2000M tok | 2287.0 min | val_loss 2.8962
  - sample: '“The U.N. has the most comprehensive data on the human cost of disease, with a total cost of $4.5 trillion more than the world health budget.” — Rachel Carson\nGlobal health costs and spending trends have improved dramatically in the last 20 years and are now at an all-time high. In 2009, annual health costs were $6.8 trillion and global health spending was $5.5 trillion. The new UN Intergovernmental Committee on Climate Change released its annual report'

### Capstone — mini_220m (217M, tied embeddings + Muon, gradient checkpointing)
Applied the 3-seed-confirmed lab winner (tied embeddings) at scale, fitting 217M on 8 GB with gradient checkpointing.
- **mini_220m [muon]** (2026-08-20) | 217.1M | 3815 iters, 2.0B tok | 2287 min (~38h) | val_loss **2.8962**
Beats the 125M base (2.9687) on **2B tokens vs 3B** — the scaling + recipe sample-efficiency win (1 clean resume mid-run, no loss).

Benchmarks (0-shot, lm-evaluation-harness):
| task | 217M | our 125M | GPT-2 124M | Pythia-160m | chance |
|---|---|---|---|---|---|
| ARC-Easy (acc) | **53.6** | 52.2 | ~43.5 | ~44 | 25 |
| HellaSwag (acc_norm) | **33.0** | 31.9 | ~31.1 | ~30.5 | 25 |
| PIQA (acc) | 62.5 | 62.2 | ~62.9 | ~62 | 50 |
| LAMBADA (acc) | 26.3 | 25.5 | ~32.6 | ~33 | 0 |
Improves on the 125M across all 4 (modest, +0.3 to +1.4 pts) despite 1/3 less data — logarithmic scaling returns, confirmed. Beats GPT-2/Pythia on ARC-Easy + HellaSwag; LAMBADA still the weak spot (24k custom vocab + short 2B-token training). Overlay: runs/capstone_loss.png.
