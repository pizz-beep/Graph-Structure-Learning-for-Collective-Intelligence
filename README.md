# Graph Structure Learning for Collective Intelligence

**Learning sparse communication graphs end-to-end for collective decision making under uncertainty.**

> N agents. Each sees only noisy local observations. No graph is given.  
> We learn which agents should communicate — jointly with the task.

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start (Colab)

See `notebooks/train_gsl.ipynb` — open directly in Colab.

## Project Structure

```
src/gsl/
  model.py        # GSL + GNN model
  layers.py       # Graph structure learner layer
  loss.py         # Joint task + sparsity loss
  data.py         # METR-LA / synthetic loaders
  train.py        # Training loop
  evaluate.py     # Recovery metrics + visualization
configs/
  metrla.yaml     # Experiment config
  synthetic.yaml
experiments/
  run_ablation.py # Fixed random / kNN / learned comparison
notebooks/
  train_gsl.ipynb # Main Colab notebook
```

## Key Results (to reproduce)

1. Learned graph recovers ground-truth sensor topology without supervision
2. Outperforms fixed kNN and random graph baselines on anomaly detection
3. Interpretable adjacency visualizations showing discovered neighborhoods
