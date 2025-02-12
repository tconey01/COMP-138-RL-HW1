# Reinforcement Learning - Nonstationary Multi-Armed Bandit Experiment  

## **Overview**  
This repository contains an implementation of a **nonstationary k-armed bandit problem** to compare different **reinforcement learning strategies**. The experiment evaluates how the **ε-greedy strategy** affects the performance of:
1. **Sample-average action-value method**  
2. **Constant step-size action-value method (α = 0.1)**  

The experiment is conducted across multiple **ε values (0.01, 0.1, 0.5)** to analyze their effect in a **nonstationary environment**.  

---

## **Files in this Repository**  

| File | Description |
|------|------------|
| `bandit.py` | Contains the `NonstationaryBandit` class and `run_experiment` function. |
| `main.py` | Runs the original reinforcement learning experiment without modifying **ε-greedy behavior**. |
| `main_experiment.py` | Runs an **extended experiment** to evaluate different **ε values (0.01, 0.1, 0.5)**. |
| `results/` | Directory where output plots are saved. |
| `README.md` | Instructions on setting up and running the experiment. |

---

## **Installation & Setup**  

Ensure you have **Python 3.x** installed. Install required dependencies using:  
```bash
pip install numpy matplotlib tqdm
