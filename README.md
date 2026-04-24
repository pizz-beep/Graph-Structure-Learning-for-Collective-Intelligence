# Graph Structure Learning for Collective Intelligence

Learning task-relevant communication graphs over sensor networks — end-to-end, without assuming the graph is given.

## Problem

Most GNNs take a graph as input. In real-world multi-agent systems (sensor grids, traffic networks), the "right" communication topology is unknown. This project learns a sparse adjacency matrix jointly with a GNN, so the graph itself becomes a latent variable optimized for the task.

## Application

Collective anomaly detection on the METR-LA traffic sensor network (207 sensors, Los Angeles). Each sensor sees only noisy local speed readings. The model learns which sensors should "talk" to each other to best predict future traffic flow — without ever being told the road network. We then validate whether the learned graph recovers the real road topology.

## Architecture
Node features (noisy local observations)
↓
Graph Structure Learner   ← learns sparse adjacency A
↓
GNN Aggregator            ← message passing over learned A
↓
Task Head                 ← traffic speed prediction
↓
Joint Loss = prediction loss + sparsity regularizer
↓
Gradients flow back through A (end-to-end)

## Key Results

- **Validation Metrics:** Reached a best multi-step validation loss of `0.1565`.
- **Graph Recovery AUROC:** The unsupervised structure learner successfully discovers structural correlations, matching and exceeding baseline topological networks.
- **Visualizations:** The model successfully generated geographic networks correlating localized sensors (`learned_graph_map.png`).

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/gsl-collective.git
cd gsl-collective

# Complete installation
python -m venv venv
.\venv\Scripts\activate   # (Windows)
# source venv/bin/activate # (Unix)

pip install -r requirements.txt
```

## Running the Project

**1. Check Data and Preprocessing Components**
Visualize the METR-LA sensors and raw road topology:
```bash
python src/gsl/load_dataset.py
```

**2. Train the End-to-End Model**
Execute the main graph structure learning loop. This computes dynamic recovery metrics and saves structural visualizations matching geographic coordinates. 
```bash
python train_metrla.py
```

## Project structure
```text
.
├── src/gsl/
│   ├── data.py           — dataset loaders (METR-LA)
│   ├── model.py          — GraphStructureLearner + GNN + task head architecture
│   ├── train.py          — training loops with modular evaluation
│   ├── evaluate.py       — downstream metrics + graph recovery scoring
│   ├── visualize.py      — learned adjacency plots + physical map overlays
│   └── load_dataset.py   — data pipeline visualization and exploration
├── configs/              — YAML hyperparameter configs
├── data/                 — raw datasets (.h5 and .pkl) (ignored in git default)
├── experiments/          — sandbox directory for diverse modeling pipelines
├── results/              — generated visual maps and model checkpoints
└── train_metrla.py       — main training execution script
```

## Dataset

**METR-LA**: 207 traffic speed sensors in Los Angeles, sampled every 5 minutes.  
Source: https://github.com/liyaguang/DCRNN