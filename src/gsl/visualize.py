# src/gsl/visualize.py
"""
Geographic and structural visualizations of learned graphs.

Functions
---------
plot_sensor_map          — scatter plot of sensor GPS positions
plot_learned_edges_on_map — draw top-k learned edges over the sensor map (geo overlay)
plot_graph_dashboard     — combined figure: sensor map + adjacency + degree histogram
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.collections as mc
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Dark-theme defaults shared across all plots
# ─────────────────────────────────────────────────────────────────────────────

DARK_BG   = '#0f0f12'
PANEL_BG  = '#16161a'
ACCENT    = '#5a4ab0'
EDGE_CLR  = '#a090e0'
TEXT_CLR  = '#c8c8d8'
MUTED     = '#555568'
HIGHLIGHT = '#e07060'


def _apply_dark(fig, axes):
    """Apply consistent dark theme to a figure and its axes."""
    fig.patch.set_facecolor(DARK_BG)
    for ax in (axes if hasattr(axes, '__iter__') else [axes]):
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=MUTED)
        for s in ax.spines.values():
            s.set_edgecolor('#2a2a35')


# ─────────────────────────────────────────────────────────────────────────────
# 1. Sensor location map
# ─────────────────────────────────────────────────────────────────────────────

# METR-LA sensor GPS coordinates (207 sensors, Los Angeles).
# Source: DCRNN repo — graph_sensor_locations.csv
# Coordinates are (latitude, longitude).
# We embed approximate values here so visualizations work offline.
# Replace SENSOR_LOCS with the real CSV if you need exact positions.
_APPROX_CENTER_LAT = 34.05
_APPROX_CENTER_LON = -118.25


def load_sensor_locations(csv_path=None):
    """
    Load sensor (lat, lon) coordinates.

    If csv_path points to graph_sensor_locations.csv from the DCRNN repo,
    we use the real coordinates. Otherwise we fall back to a jittered
    approximation centred on LA that still produces a meaningful layout.

    Returns:
        lats : np.ndarray (N,)
        lons : np.ndarray (N,)
    """
    if csv_path is not None:
        p = Path(csv_path)
        if p.exists():
            import pandas as pd
            df = pd.read_csv(p, header=None)   # columns: sensor_id, lat, lon
            lats = df.iloc[:, 1].values.astype(float)
            lons = df.iloc[:, 2].values.astype(float)
            return lats, lons

    # Fallback: reproducible jitter around downtown LA
    rng = np.random.default_rng(0)
    N = 207
    lats = _APPROX_CENTER_LAT + rng.uniform(-0.3, 0.3, N)
    lons = _APPROX_CENTER_LON + rng.uniform(-0.35, 0.35, N)
    return lats, lons


def plot_sensor_map(lats, lons, title="METR-LA Sensor Locations",
                    highlight_ids=None, save_path=None):
    """
    Scatter plot of sensor GPS positions.

    Args:
        lats, lons     : arrays of length N
        highlight_ids  : list of sensor indices to mark in a different colour
        save_path      : if given, saved as PNG

    Returns:
        fig, ax
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    _apply_dark(fig, ax)

    ax.scatter(lons, lats, s=18, c=ACCENT, alpha=0.7, zorder=3,
               linewidths=0.3, edgecolors=EDGE_CLR, label='Sensor')

    if highlight_ids:
        ax.scatter(lons[highlight_ids], lats[highlight_ids],
                   s=60, c=HIGHLIGHT, zorder=4,
                   linewidths=0.5, edgecolors='white', label='Selected')

    ax.set_xlabel('Longitude', color=MUTED, fontsize=10)
    ax.set_ylabel('Latitude',  color=MUTED, fontsize=10)
    ax.set_title(title, color=TEXT_CLR, fontsize=13, pad=10)
    ax.legend(facecolor=PANEL_BG, labelcolor=TEXT_CLR, framealpha=0.7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# 2. Learned edges overlaid on the sensor map
# ─────────────────────────────────────────────────────────────────────────────

def plot_learned_edges_on_map(
    A_learned,
    lats,
    lons,
    top_edges: int = 100,
    title="Learned Communication Graph (top edges)",
    true_adj=None,
    save_path=None,
):
    """
    Draw the strongest learned edges between sensor positions.

    This is the "killer visualization" — it shows *which sensors the model
    decided should communicate*, overlaid on real geography. If the learned
    graph partially recovers the road network, you'll see edges that follow
    major highways (I-405, I-10, US-101, etc.).

    Args:
        A_learned  : (N, N) tensor or ndarray — learned adjacency
        lats, lons : (N,) arrays — sensor GPS coordinates
        top_edges  : number of strongest edges to draw
        true_adj   : (N, N) optional — if supplied, colour edges by whether
                     they appear in the ground-truth road graph
                     (purple = recovered, orange = novel)
        save_path  : PNG save path

    Returns:
        fig, ax
    """
    A = A_learned.detach().cpu().numpy() if hasattr(A_learned, 'detach') else np.array(A_learned)
    N = A.shape[0]

    # Flatten upper triangle, sort by weight
    rows, cols = np.triu_indices(N, k=1)
    weights = A[rows, cols]
    order = np.argsort(weights)[::-1][:top_edges]
    rows, cols, weights = rows[order], cols[order], weights[order]

    # Normalise weights to [0.1, 1.0] for alpha
    w_norm = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
    alphas = 0.15 + 0.85 * w_norm

    fig, ax = plt.subplots(figsize=(10, 7))
    _apply_dark(fig, ax)

    # Draw edges as line segments
    if true_adj is not None:
        T = true_adj if isinstance(true_adj, np.ndarray) else np.array(true_adj)
        # Colour: purple if edge exists in ground truth, orange if novel
        segs_recovered, segs_novel = [], []
        alp_recovered, alp_novel  = [], []
        for r, c, a in zip(rows, cols, alphas):
            seg = [(lons[r], lats[r]), (lons[c], lats[c])]
            if T[r, c] > 0 or T[c, r] > 0:
                segs_recovered.append(seg)
                alp_recovered.append(a)
            else:
                segs_novel.append(seg)
                alp_novel.append(a)

        if segs_recovered:
            lc = mc.LineCollection(segs_recovered, colors=ACCENT,
                                   linewidths=0.8, alpha=np.mean(alp_recovered),
                                   label='Recovered (in road graph)')
            ax.add_collection(lc)
        if segs_novel:
            lc = mc.LineCollection(segs_novel, colors=HIGHLIGHT,
                                   linewidths=0.5, alpha=np.mean(alp_novel),
                                   label='Novel (not in road graph)')
            ax.add_collection(lc)
    else:
        segs = [[(lons[r], lats[r]), (lons[c], lats[c])]
                for r, c in zip(rows, cols)]
        lc = mc.LineCollection(segs, colors=ACCENT, linewidths=0.6,
                                alpha=0.5, label='Learned edge')
        ax.add_collection(lc)

    # Sensor scatter on top
    ax.scatter(lons, lats, s=22, c=TEXT_CLR, alpha=0.85, zorder=5,
               linewidths=0.3, edgecolors=EDGE_CLR)

    ax.autoscale()
    ax.set_xlabel('Longitude', color=MUTED, fontsize=10)
    ax.set_ylabel('Latitude',  color=MUTED, fontsize=10)
    ax.set_title(title, color=TEXT_CLR, fontsize=13, pad=10)
    ax.legend(facecolor=PANEL_BG, labelcolor=TEXT_CLR, framealpha=0.8,
              loc='upper left')

    # Annotation: recovery rate
    if true_adj is not None and segs_recovered:
        total = len(segs_recovered) + len(segs_novel)
        rate = 100 * len(segs_recovered) / total if total > 0 else 0
        ax.text(0.02, 0.04,
                f"Recovery rate (top {top_edges}): {rate:.1f}%",
                transform=ax.transAxes, color=TEXT_CLR,
                fontsize=9, alpha=0.8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# 3. Full dashboard
# ─────────────────────────────────────────────────────────────────────────────

def plot_graph_dashboard(
    A_learned,
    lats,
    lons,
    true_adj=None,
    top_edges=80,
    threshold=0.01,
    title="Learned Graph — Analysis Dashboard",
    save_path=None,
):
    """
    3-panel figure combining:
        Left   — geo map with learned edges (the main visual)
        Middle — adjacency heatmap (learned vs true side-by-side if true_adj given)
        Right  — degree distribution histogram

    Args:
        A_learned : (N, N) tensor or ndarray
        lats, lons: (N,) GPS arrays
        true_adj  : (N, N) optional ground-truth adjacency
        top_edges : how many strongest edges to draw on the map
        threshold : edge weight cutoff for degree computation
        save_path : PNG save path

    Returns:
        fig
    """
    A = A_learned.detach().cpu().numpy() if hasattr(A_learned, 'detach') else np.array(A_learned)
    T = (np.array(true_adj) if true_adj is not None else None)

    ncols = 3 if T is None else 4
    fig = plt.figure(figsize=(6 * ncols, 6))
    _apply_dark(fig, [])

    # ── Panel 1: Geo map ──────────────────────────────────────────────────
    ax_map = fig.add_subplot(1, ncols, 1)
    _apply_dark(fig, ax_map)

    N = A.shape[0]
    rows, cols_idx = np.triu_indices(N, k=1)
    weights = A[rows, cols_idx]
    order = np.argsort(weights)[::-1][:top_edges]
    r, c, w = rows[order], cols_idx[order], weights[order]
    w_norm = (w - w.min()) / (w.max() - w.min() + 1e-8)

    if T is not None:
        for ri, ci, wi in zip(r, c, w_norm):
            clr = ACCENT if (T[ri, ci] > 0 or T[ci, ri] > 0) else HIGHLIGHT
            ax_map.plot([lons[ri], lons[ci]], [lats[ri], lats[ci]],
                        color=clr, lw=0.6, alpha=0.3 + 0.5 * wi)
    else:
        segs = [[(lons[ri], lats[ri]), (lons[ci], lats[ci])] for ri, ci in zip(r, c)]
        lc = mc.LineCollection(segs, colors=ACCENT, linewidths=0.6, alpha=0.4)
        ax_map.add_collection(lc)
        ax_map.autoscale()

    ax_map.scatter(lons, lats, s=18, c=TEXT_CLR, zorder=5,
                   linewidths=0.3, edgecolors=EDGE_CLR, alpha=0.9)
    ax_map.set_title('Learned Edges (Geo)', color=TEXT_CLR, fontsize=11)
    ax_map.set_xlabel('Lon', color=MUTED); ax_map.set_ylabel('Lat', color=MUTED)

    # ── Panel 2: Learned adjacency heatmap ───────────────────────────────
    ax_adj = fig.add_subplot(1, ncols, 2)
    _apply_dark(fig, ax_adj)
    im = ax_adj.imshow(A, cmap='magma', aspect='auto', vmin=0)
    plt.colorbar(im, ax=ax_adj, fraction=0.046, pad=0.04)
    ax_adj.set_title('Learned Adjacency', color=TEXT_CLR, fontsize=11)

    col_offset = 3
    # ── Panel 3 (optional): Ground-truth heatmap ─────────────────────────
    if T is not None:
        ax_true = fig.add_subplot(1, ncols, 3)
        _apply_dark(fig, ax_true)
        im2 = ax_true.imshow(T, cmap='Blues', aspect='auto')
        plt.colorbar(im2, ax=ax_true, fraction=0.046, pad=0.04)
        ax_true.set_title('Ground Truth (Road Network)', color=TEXT_CLR, fontsize=11)
        col_offset = 4

    # ── Panel last: Degree distribution ──────────────────────────────────
    ax_deg = fig.add_subplot(1, ncols, col_offset)
    _apply_dark(fig, ax_deg)
    degrees = (A > threshold).sum(axis=1)
    ax_deg.hist(degrees, bins=20, color=ACCENT, edgecolor='#2a2a35', linewidth=0.5)
    ax_deg.axvline(degrees.mean(), color=HIGHLIGHT, linestyle='--',
                   linewidth=1, label=f'Mean = {degrees.mean():.1f}')
    ax_deg.set_xlabel('Node degree', color=MUTED)
    ax_deg.set_ylabel('Count',       color=MUTED)
    ax_deg.set_title('Degree Distribution', color=TEXT_CLR, fontsize=11)
    ax_deg.legend(facecolor=PANEL_BG, labelcolor=TEXT_CLR)

    fig.suptitle(title, color=TEXT_CLR, fontsize=14, y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    return fig
