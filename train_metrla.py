# save as train_metrla.py in the project root, then run:
# python train_metrla.py

import sys, yaml, torch
sys.path.insert(0, 'src')
from gsl.data import build_dataloaders
from gsl.model import GSLNet
from gsl.loss import GSLLoss
from gsl.train import train
from gsl.evaluate import graph_recovery_auroc, regression_metrics
from gsl.visualize import load_sensor_locations, plot_learned_edges_on_map
import pickle, numpy as np

# Config
cfg = yaml.safe_load(open('configs/metrla.yaml'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Data
train_loader, val_loader, test_loader, scaler, adj_true = build_dataloaders(
    h5_path=None, adj_dir=None,
    in_steps=12, out_steps=3,
    batch_size=cfg['batch_size'],
)

# Model — in_features = 12 timesteps per sensor
model = GSLNet(
    in_features=12,
    hidden_dim=cfg['hidden_dim'],
    num_classes=3,          # predicting 3 future steps
    gnn_layers=cfg['gnn_layers'],
    top_k=cfg['top_k'],
)

criterion = GSLLoss(
    lambda_s=cfg['sparsity_lambda'],
    lambda_sm=0.01,
    lambda_c=0.001,
)
optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])

# Train
model = train(model, train_loader, val_loader, criterion, optimizer, cfg, device)

# Evaluate graph recovery — THE KILLER RESULT
model.eval()
with torch.no_grad():
    x_batch, _ = next(iter(test_loader))
    _, A_learned, _ = model(x_batch.to(device))
    A_learned = A_learned.mean(0).cpu()   # average over batch

adj_true_t = torch.tensor(adj_true)
auroc = graph_recovery_auroc(A_learned, adj_true_t)
print(f'\nGraph recovery AUROC: {auroc:.4f}   (random chance = 0.50)')

# Visualize on geo map
lats, lons = load_sensor_locations()
fig = plot_learned_edges_on_map(
    A_learned, lats, lons,
    top_edges=150,
    true_adj=adj_true_t,
    save_path='results/learned_graph_map.png',
)
print('Saved: results/learned_graph_map.png')
