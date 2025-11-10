# RIZE: Adaptive Regularization for Imitation Learning
  
**Paper**: [TMLR 2025](https://openreview.net/forum?id=a6DWqXJZCZ) | [arXiv](https://arxiv.org/abs/2502.20089)

---

RIZE is a novel IRL method with **adaptive reward bounds** and **distributional critics**, achieving expert-level performance on MuJoCo and Adroit tasks using only 3–10 expert demonstrations. It outperforms IQ-Learn, LSIQ, SQIL, CSIL, and BC—especially on high-DoF tasks like Humanoid-v2 and Hammer-v1. See the [project page](https://adibka.github.io/RIZE/) for results, and visuals.

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/adibka/RIZE.git
   cd RIZE
   ```

2. **Create an environment (Python 3.10)**
   ```bash
   python3 -m venv venvs/rize
   source venvs/rize/bin/activate
   ```

3. **Install mujoco-py**

   > **Note**: `mujoco-py==2.0.2.0` requires MuJoCo version 2.0 binaries. Follow the guide at [mujoco-py docs](https://github.com/openai/mujoco-py/tree/v2.0.2.0) if needed.
   
   ```bash
   mkdir -p ~/.mujoco && cd ~/.mujoco
   curl -O https://www.roboti.us/download/mujoco200_linux.zip
   curl -O https://www.roboti.us/file/mjkey.txt
   unzip mujoco200_linux.zip
   mv mujoco200_linux mujoco200
   rm mujoco200_linux.zip
   ```

   Add to `~/.bashrc` (replace `{user-name}` with your actual username):
   ```bash
   echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/{user-name}/.mujoco/mujoco200/bin' >> ~/.bashrc
   source ~/.bashrc
   ```
   
4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   > **Note**: `requirements.txt` pins tested versions (including `mujoco-py`, `gym`, `gymnasium`, `mujoco`). Verified on Ubuntu 22.04 + Python 3.10.

---

## Expert Demonstrations

Download pre-generated expert trajectories:
```bash
cd RIZE
./download_experts.sh
```

---

## Training

Run training with a single command. Hyperparameters are detailed in the **paper appendix** (Table 1).

```bash
python main.py --env halfcheetah --demos 10 --seed 0
```

> Replace `--env` with any supported task:  
> `halfcheetah`, `walker2d`, `ant`, `humanoid`, `hopper`, `hammer`  

---

## Citation

```bibtex
@article{karimi2025rize,
  title={RIZE: Adaptive Regularization for Imitation Learning},
  author={Karimi, Adib and Ebadzadeh, Mohammad Mehdi},
  journal={Transactions on Machine Learning Research},
  year={2025},
  url={https://openreview.net/forum?id=a6DWqXJZCZ}
}
```

---

