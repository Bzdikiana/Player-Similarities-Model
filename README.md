# ⚽ Finding the Next Ramos (Because Defending is an Art)

> *"The best defenders don't just stop attacks—they read the game three moves ahead."*

## The Problem That Kept Me Up at Night

As someone who played center back, nothing frustrates me more than how people compare defenders.

"Oh, Ramos and Van Dijk are both world-class center backs." Sure. But watch them actually play and they're **completely different footballers**.

Ramos steps up, presses high, wins the ball in midfield, plays like a midfielder who happens to defend. Van Dijk sits deep, reads the game, times his challenges perfectly, rarely commits. Both elite. Totally different decision-making.

And it's not just those two. Think about it:

- **High line vs deep block** - Does the CB push up to squeeze space or drop to protect the box?
- **Aggressive press vs patient reading** - Jump out to win the ball early or wait for the attacker to commit?
- **Aerial dominance vs ground duels** - Attack every cross or stay on feet and sweep up?
- **Ball-playing vs direct** - Look for the progressive pass or hit it long and safe?
- **Cover shadow vs man-marking** - Protect zones or stick tight to a runner?

Traditional stats say "tackles won" and "aerial duels." They don't tell you *when* a defender chooses to engage, *how* they position before a ball is even played, or *why* they step out instead of holding the line.

That's what this project tries to capture.

---

## What This Actually Does

I built a deep learning model that watches how players make decisions in different game situations and learns to identify players who *think* alike on the pitch.

**The core idea:**
> Two players are similar if, when faced with the same situation, they'd make the same choice.

Not "they both make 5 clearances per game" but "when a winger receives the ball wide and shapes to cross, do they step out to press or hold position and track the runner?"

### How?

1. **Graph Neural Networks** look at each moment - where's the attacker? where's the covering midfielder? is there space behind?
2. **Transformers** track patterns over time - does this defender always step out early? do they stay compact under pressure?
3. **Contrastive Learning** pulls similar decision-makers together in embedding space

The result: a 64-dimensional fingerprint of how each player defends (or attacks, or creates, depending on position).

---

## The Data

Using StatsBomb's 360 data (freeze-frames showing where every player is at each moment):

- 🏆 **World Cup 2022** - High-pressure international football
- ⚽ **La Liga 2020/21** - Technical, possession-heavy defending
- 🇩🇪 **Bundesliga 2023/24** - Transition chaos and counter-pressing
- 🇫🇷 **Ligue 1** - Physical battles and tactical setups
- 🏴󠁧󠁢󠁥󠁮󠁧󠁿 **Euros 2024 & 2020** - Tournament knockout pressure
- 🇺🇸 **MLS 2023** - Different tactical contexts

**~2,000 matches. ~6 million events. 1,000+ players.**

---

## Results (The Honest Version)

| What's Working | What's Not (Yet) |
|---------------|------------------|
| ✅ Same-position players cluster together | ⚠️ Doesn't know if the tackle was clean or reckless |
| ✅ High-line CBs vs deep-block CBs - different embeddings | ⚠️ Only works with 360 data |
| ✅ Ball-playing defenders vs clearance merchants - clearly separated | ⚠️ Need more data for rare positions |
| ✅ 55% position prediction accuracy (random would be 7%) | ⚠️ Off-ball positioning not fully captured |

---

## Get Started

Clone the repo, install requirements, and run `player_similarity_training.ipynb`. That's it.

---

## Project Structure

```
player-similarity/
├── player_similarity_training.ipynb  # The main event 🎯
├── src/
│   ├── datasets/          # Data loading & graph building
│   ├── models/            # GNN + Transformer architecture
│   ├── training/          # Loss functions & training loop
│   └── retrieval/         # Similarity search API
├── outputs/               # Trained model & embeddings
├── docs/                  # Technical details for the curious
├── previous/              # Earlier experiments (the journey)
└── requirements.txt
```

---

## The Nerdy Details (For Those Who Care)

- **Architecture**: 3-layer Graph Attention Network → 2-layer Temporal Transformer → 64-dim embeddings
- **Loss**: InfoNCE contrastive loss (temperature=0.07) + position prediction auxiliary task
- **Training**: 50 epochs, batch size 32, Adam with lr=1e-4
- **Final metrics**: Loss 0.83, Similarity Gap 0.624

Full math in [docs/MATH_FOUNDATIONS.md](docs/MATH_FOUNDATIONS.md) if you're into that sort of thing.

---

## What I'd Do With More Time

1. **Defender archetypes** - Cluster CBs into "aggressive press" vs "deep reader" vs "ball-player"
2. **Tactical context** - How does a defender adapt when playing high line vs low block?
3. **Scout interface** - "Find me a left-footed CB under 25 who can play out from the back"
4. **Outcome tracking** - Did that stepout win the ball or get beaten?

---

## Contributing

Found a bug? Have an idea? Open an issue or PR.

If you're a scout and this helped you find a hidden gem, I accept compensation in the form of match tickets.

---

## Acknowledgments

- **StatsBomb** for the incredible open data (seriously, this wouldn't exist without them)
- The GNN/Transformer research community for the architectures
- Every CB who ever had to deal with a winger cutting inside—I feel you

---

## Contact

**Armen Bzdikian**  
📧 bzdikiana11@gmail.com

*Built with too much coffee and a deep love for the beautiful game.*
