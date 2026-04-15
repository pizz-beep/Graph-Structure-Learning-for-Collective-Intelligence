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

## Key Results (to be filled after experiments)

- Learned graph vs. random graph vs. kNN graph (ablation)
- Graph recovery: does the learned adjacency match the road network?
- Visualization: which sensor connections did the model discover?

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/gsl-collective.git
cd gsl-collective
pip install -r requirements.txt
```

## Running experiments

Open any notebook in `experiments/` in Google Colab. Each notebook pulls the latest code from GitHub and runs one experiment end-to-end.

## Project structure
src/gsl/
data.py        — dataset loaders (METR-LA)
model.py       — GraphStructureLearner + GNN + task head
train.py       — training loop with W&B logging
evaluate.py    — prediction metrics + graph recovery score
visualize.py   — learned adjacency plots + map overlays
experiments/     — Colab notebooks (one per experiment)
configs/         — YAML hyperparameter files
data/            — raw + processed datasets (gitignored)

## Dataset

METR-LA: 207 traffic speed sensors in Los Angeles, sampled every 5 minutes.  
Source: https://github.com/liyaguang/DCRNN