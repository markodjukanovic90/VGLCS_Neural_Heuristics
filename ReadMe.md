# Project Overview

This document summarizes the project structure, main components, and acknowledgements for the paper:

**"Neuro-Evolved Heuristics for Variable Gapped Common Subsequence Identification" (Djukanovic et al.)**

---

# 1. Project Directory Structure

The project is organized as follows:

```
project-root/
│
├── Eugen/          # Core experimental framework and evolutionary components
├── results/        # Experimental outputs, logs, and analysis results
├── software/       # Implementation of algorithms, heuristics, and models
├── docs/           # Documentation, figures, and paper-related materials
├── instances/      # Benchmark instances used in experiments
└── README.md
```

---

# 2. Project Description

This project investigates **neuro-evolved heuristics** for solving the Variable Gapped Common Subsequence (VGCS) / related combinatorial optimization problems. The main focus is on:

* Evolutionary and metaheuristic optimization (GA strategies)
* Neural-guided decision-making for heuristic improvement
* Beam / search-based constructive methods
* Hybrid ILP / heuristic / learning-based approaches
* Benchmark evaluation on structured instance sets

---

# 3. Core Components

### Eugen Module

Contains evolutionary frameworks and neuro-evolution strategies used to guide heuristic construction and selection.

### Software Module

Implements all core algorithms including:

* Beam Search variants (hand-crafted, NN-based score, ensembe method)
* Sotware for training Neural Network-based heuristic guidances

### Instances

Contains benchmark datasets for VGCS-related problems, used for evaluation and comparison of algorithms.

### Results

Stores experimental outputs including:

* Solution quality metrics
* Runtime statistics
* Comparative performance tables
* Figures

### Docs

Includes:

* Paper-related figures and TikZ diagrams
* Technical documentation
* Supplementary materials

---

# 4. Experimental Pipeline

1. Load instance from `instances/`
2. Run baseline heuristics from `software/`
3. Apply neuro-evolved heuristic strategies from `Eugen/`
4. Collect and store outputs in `results/`

---

# 5. Acknowledgements

This publication is co-funded by the European Union’s Horizon Europe research and innovation program under the Marie Skłodowska-Churie COFUND Postdoctoral Programme grant agreement No. 101081355 — SMASH, and by the Republic of Slovenia and the European Union from the European Regional Development Fund. Co-funded by the European Union.

Views and opinions expressed are those of the authors only and do not necessarily reflect those of the European Union or the European Research Executive Agency (REA). Neither the European Union nor the REA can be held responsible for them.

The authors gratefully acknowledge the SLING consortium for funding this research by providing computing resources of the HPC Vega at the Institute of Information Science ([www.izum.si](http://www.izum.si)).

---

# 6. Notes

* Designed for reproducibility and HPC execution (e.g., SLURM systems)
* Modular structure allows extension to other sequence-based optimization problems
* Supports integration of neuro-evolutionary and classical heuristic frameworks

```
```




