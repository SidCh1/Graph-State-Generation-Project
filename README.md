# Graph-State-Generation-Project
This is the complete code for the paper: https://arxiv.org/abs/2412.04252

# Networkx and standard python libraries are used.


# How to use this code?

First all files should be in same directory. 

The file calculate_gates_bell_paris_main.py contains all the functions required to get data and plots that are shown in the paper that has gates and bell pair analysis. The file Run_statistics.ipynb must be opened and run. That will create all the data. 

For Bell pair sources, you have to run the following files. ER is for erdos renyi and BA is for barabasi albert. 

Bell_Pair_Sources_fix_P_ER.py

Bell_Pair_Sources_fix_C_BA.py

We have also studied using Bell pairs or GHZs as building blocks in star topology. That can be found in the file Bell_Pair_vs_GHZ_Building_Block.py


Here’s a cleaner, well-formatted version suitable for a GitHub README (`.md`) file:

---

# Graph-State-Generation-Project

This repository contains the complete code used in the paper:
🔗 [https://arxiv.org/abs/xx](https://arxiv.org/abs/xx)

---

## 📦 Dependencies

The project uses:

* `networkx`
* Standard Python libraries

Make sure these are installed before running the code.

---

## 🚀 Getting Started

1. Ensure that **all files are placed in the same directory**.
2. Open and run the notebook:

   ```
   Run_statistics.ipynb
   ```

   This will generate all the datasets used in the paper.

---

## 📊 Main Script

The file:

```
calculate_gates_bell_paris_main.py
```

contains all the core functions required to:

* Generate data
* Reproduce plots from the paper
* Perform gate and Bell pair analysis

---

## 🔗 Bell Pair Source Simulations

To simulate Bell pair sources, run the following scripts:

* **Erdős–Rényi (ER) model:**

  ```
  Bell_Pair_Sources_fix_P_ER.py
  ```

* **Barabási–Albert (BA) model:**

  ```
  Bell_Pair_Sources_fix_C_BA.py
  ```

---

## ⭐ Additional Study: Bell Pairs vs GHZ States

We also investigate the use of Bell pairs and GHZ states as building blocks in a star topology.

This analysis is implemented in:

```
Bell_Pair_vs_GHZ_Building_Block.py
```

---

If you want, I can also add badges, installation steps (`pip install`), or a cleaner project structure section.
