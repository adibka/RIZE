# 🚀 **RIZE**: ***Regularized Imitation Learning via Distributional Reinforcement Learning***

---

## 📖 Summary

*RIZE* is a novel inverse reinforcement learning method that enhances ***Maximum Entropy IRL*** by incorporating a squared temporal-difference regularizer with adaptive, dynamically evolving targets, enabling more stable and flexible reward learning. It integrates distributional reinforcement learning to capture richer return distributions, improving value function representation. ***RIZE*** achieves state-of-the-art results on challenging continuous control benchmarks in ***MuJoCo***, including expert-level performance on the ***Humanoid*** task with only three demonstrations.

---

## ⚙️ Installation

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
---

## ▶️ Usage

To reproduce the results from our paper for **HalfCheetah-v2** with **Seed 0**, run:

```bash
python train.py --env halfcheetah --seed 0
```
---

## 📊 Results

Below are main results from our experiments:

<p align="center">
  <img src="images/bar.png" alt="Normalized return of RIZE vs. online imitation learning baselines on Gym MuJoCo tasks." width="750"/>
</p>
<p align="center"><em>Figure: Normalized return of RIZE vs. online imitation learning baselines on Gym MuJoCo tasks. We depict the sorted top 25% episodic returns across five seeds to evaluate convergence to expert-level behavior. We evaluate with three and ten expert trajectories.</em></p>


<p align="center">
  <img src="images/main_curves_demo3&10.png" alt="Normalized return of RIZE vs. online imitation learning baselines on Gym MuJoCo tasks." width="750"/>
</p>
<p align="center"><em>Figure: Normalized return of RIZE vs. online imitation learning baselines on Gym MuJoCo tasks. n
denotes the number of expert trajectories.</em></p>

---

## 🙏 Acknowledgements

We acknowledge the following repositories for their contributions to our work:

- https://github.com/Div99/IQ-Learn
- https://github.com/rail-berkeley/rlkit
- https://github.com/xtma/dsac
- https://github.com/robfiras/ls-iq/tree/main

---

