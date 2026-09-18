# Think First, Know Later

Official repository for the paper:

**"Think First, Know Later: Temporal Confidence Signals in Large Language Models"**

This project studies temporal confidence estimation in large language models by comparing:

- **Pre-solve confidence** (*Feeling-of-Knowing / FOK*)
- **Post-solve confidence** (*Judgment-of-Learning / JOL*)

We evaluate how confidence changes before and after reasoning across mathematical reasoning, logical reasoning, and factual recall tasks.

---

## Repository Structure

```text
.
|-- core/    # Core experimentation and evaluation pipeline
`-- data/    # Benchmark datasets used in experiments
```

### `core/`
Contains (per domain):
- Confidence prompting pipeline
- Evaluation scripts

### `data/`
Contains the three benchmark datasets used in the paper:
- RealMath
- RiddleBench
- TriviaQA

---

## Main Idea

Most prior work studies confidence only *after* answer generation.

This project introduces a temporal framing of confidence:
- Can models estimate correctness **before reasoning begins?**
- Does confidence improve **after reasoning is generated?**
- Are hallucinations associated with persistently high confidence?

---

## Citation

If you use this repository, please cite the accompanying paper.

```bibtex
@article{kale2026thinkfirst,
  title={Think First, Know Later: Temporal Confidence Signals in Large Language Models},
  author={Kale, Sahil},
  year={2026}
}
```
