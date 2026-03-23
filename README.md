Here’s your README with a clean citation section and BibTeX added:

---

# 📡 Graph-State-Generation-Project

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Research-orange.svg)]()

This repository contains the complete code used in the paper:
🔗 [[https://arxiv.org/abs/2412.04252](https://arxiv.org/abs/2412.04252)]

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Graph-State-Generation-Project.git
cd Graph-State-Generation-Project
```

Install required dependencies:

```bash
pip install networkx matplotlib numpy jupyter
```

---

## 🧰 Dependencies

* `networkx`
* `numpy`
* `matplotlib`
* `jupyter`
* Standard Python libraries

---

## 📁 Project Structure

```
.
├── calculate_gates_bell_paris_main.py   # Core functions for data & plots
├── Run_statistics.ipynb                 # Main notebook to generate results
├── Bell_Pair_Sources_fix_P_ER.py        # ER (Erdős–Rényi) simulations for Bell-pair sources
├── Bell_Pair_Sources_fix_C_BA.py        # BA (Barabási–Albert) simulations for Bell-pair sources
├── Bell_Pair_vs_GHZ_Building_Block.py   # Bell vs GHZ (star topology)
└── README.md
```

---

## 🚀 Usage

### 1. Setup

Ensure all files are in the **same directory**.

### 2. Generate Data

Run the notebook:

```bash
jupyter notebook Run_statistics.ipynb
```

This will:

* Generate all datasets except for Bell-pair sources.
* Reproduce figures from the paper

---

## 📊 Core Functionality

The main script:

```
calculate_gates_bell_paris_main.py
```

Provides functions to:

* Perform gate and Bell-pair analysis
* Generate plots used in the paper except for Bell-pair sources.

---

## 🔗 Bell Pair Source Simulations

Perform Bell-pair sources analysis:

### Erdős–Rényi (ER)

```bash
python Bell_Pair_Sources_fix_P_ER.py
```

### Barabási–Albert (BA)

```bash
python Bell_Pair_Sources_fix_C_BA.py
```

---

## ⭐ Bell Pairs vs GHZ States

Study of Bell pairs and GHZ states as building blocks in a **star topology**:

```bash
python Bell_Pair_vs_GHZ_Building_Block.py
```

---

## 📖 Citation

If you use this code in your research, please cite the following paper:

```bibtex
@article{yourkey202Xgraphstate,
  title   = {Title of the Paper},
  author  = {Author One and Author Two and Author Three},
  journal = {arXiv preprint arXiv:xx},
  year    = {202X},
  url     = {https://arxiv.org/abs/xx}
}
```

---

## 📝 BibTeX (Copy-Friendly)

You can also copy it directly:

```
@misc{chelluri2025resourcecomputationallyefficientprotocolmultipartite,
      title={A resource- and computationally-efficient protocol for multipartite entanglement distribution in Bell-pair networks}, 
      author={S. Siddardha Chelluri and Sumeet Khatri and Peter van Loock},
      year={2025},
      eprint={2412.04252},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2412.04252}, 
}
```

---

## 🤝 Contributing

Contributions, issues, and suggestions are welcome!
Feel free to open a pull request or issue.

---

