# 🚀 **RIZE**: ***Regularized Imitation Learning via Distributional Reinforcement Learning***
[Paper on arXiv](https://arxiv.org/abs/2502.20089)

---

## 📖 Summary

RIZE is a non-adversarial inverse reinforcement learning framework that unifies implicit reward regularization with distributional reinforcement learning. It extends IQ-Learn and LS-IQ by replacing fixed implicit reward targets with adaptive, learnable targets that evolve during training, constraining rewards to improve stability and prevent divergence.

To capture richer return information, RIZE trains quantile-based distributional critics and uses their expectations for policy updates—preserving theoretical guarantees while improving robustness in high-dimensional control. This combination yields bounded critic values, more stable policy optimization, and better sample efficiency.

On MuJoCo benchmarks, RIZE consistently outperforms strong baselines, achieving expert-level Humanoid performance with only three demonstrations—a setting where all baselines fail. Extensive ablations confirm that both adaptive regularization and distributional value functions are essential for its gains.

---

## ⚙️ Installation

   ```bash
   # 1) Clone the repo
   git clone https://github.com/adibka/RIZE.git
   cd RIZE
   
   # 2) (Optional) create & activate a venv
   python -m venv venv
   source venv/bin/activate
   
   # 3) Install dependencies 
   pip install -r requirements.txt
   pip install gdown
   
   # 4) Download experts.zip
   gdown --fuzzy "https://drive.google.com/file/d/1q-Mc0TuUBjkqPx564m4_3RxzJv0PoWIm/view?usp=sharing" -O experts.zip
   
   # 5) Unzip into the repo root 
   unzip -o experts.zip
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

