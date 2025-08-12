# 🚀 **RIZE**: ***Regularized Implicit Reward Inverse Reinforcement Learning via Distributional RL***

---

## 📖 Summary

RIZE is a non-adversarial inverse reinforcement learning framework that unifies implicit reward regularization with distributional reinforcement learning. It extends IQ-Learn and LS-IQ by replacing fixed implicit reward targets with adaptive, learnable targets that evolve during training, constraining rewards to improve stability and prevent divergence.

To capture richer return information, RIZE trains quantile-based distributional critics and uses their expectations for policy updates—preserving theoretical guarantees while improving robustness in high-dimensional control. This combination yields bounded critic values, more stable policy optimization, and better sample efficiency.

On MuJoCo benchmarks, RIZE consistently outperforms strong baselines, achieving expert-level Humanoid performance with only three demonstrations—a setting where all baselines fail. Extensive ablations confirm that both adaptive regularization and distributional value functions are essential for its gains.

---

## ⚙️ Installation

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
---

## ▶️ Usage

To reproduce the results from our paper for HalfCheetah-v2 with Seed 0, run:

```bash
python train.py --env halfcheetah --seed 0
```
---

## 🙏 Acknowledgements

We acknowledge the following repositories for their contributions to our work:

- https://github.com/Div99/IQ-Learn
- https://github.com/rail-berkeley/rlkit
- https://github.com/xtma/dsac
- https://github.com/robfiras/ls-iq/tree/main

---

