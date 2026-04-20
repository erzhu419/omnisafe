"""
Aggregate and plot results for paper: 4 methods × 3 seeds.
Generates:
  1. Training curves (reward & cost vs episode) with mean ± std across seeds
  2. Eval curves (eval reward & cost)
  3. Summary table (LaTeX)

Usage:
  python plot_paper_results.py --results_dir paper_results
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', type=str, default='paper_results')
parser.add_argument('--output_dir', type=str, default='paper_figures')
parser.add_argument('--smooth_window', type=int, default=10, help='Smoothing window for training curves')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ─── Method configs ──────────────────────────────────────────────────
METHODS = {
    'No-Action': {
        'pattern': 'no_action/no_action_*',
        'color': 'gray',
        'linestyle': '--',
        'has_training_curve': False,
    },
    'SAC': {
        'pattern': 'sac/logs/sac_v2_bus_*_seed*_paper',
        'color': '#1f77b4',
        'linestyle': '-',
        'has_training_curve': True,
    },
    'SAC-Lagrange': {
        'pattern': 'sac_lagrange/logs/sac_lagrange_bus_*_seed*_paper',
        'color': '#ff7f0e',
        'linestyle': '-',
        'has_training_curve': True,
    },
    'Ensemble-SAC-Lag': {
        'pattern': 'ensemble/logs/**/seed*',
        'color': '#2ca02c',
        'linestyle': '-',
        'has_training_curve': True,
    },
}


def smooth(data, window):
    if window <= 1 or len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


def load_seeds(base_dir, pattern, filename):
    """Load a .npy file from all seed directories matching the pattern."""
    search = os.path.join(base_dir, pattern)
    dirs = sorted(glob(search, recursive=True))
    arrays = []
    for d in dirs:
        path = d if d.endswith('.npy') else os.path.join(d, filename)
        if os.path.isdir(d):
            path = os.path.join(d, filename)
        else:
            continue
        if os.path.exists(path):
            arr = np.load(path)
            arrays.append(arr)
    return arrays


def align_and_aggregate(arrays):
    """Align arrays to the shortest length, return mean and std."""
    if not arrays:
        return None, None
    min_len = min(len(a) for a in arrays)
    aligned = np.stack([a[:min_len] for a in arrays])
    return aligned.mean(axis=0), aligned.std(axis=0)


# ─── Load data ───────────────────────────────────────────────────────
results = {}

for method_name, cfg in METHODS.items():
    results[method_name] = {}

    if cfg['has_training_curve']:
        # Training rewards
        reward_arrays = load_seeds(args.results_dir, cfg['pattern'], 'rewards.npy')
        cost_arrays = load_seeds(args.results_dir, cfg['pattern'], 'cost.npy')
        eval_r_arrays = load_seeds(args.results_dir, cfg['pattern'], 'eval_mean_rewards.npy')
        eval_c_arrays = load_seeds(args.results_dir, cfg['pattern'], 'eval_mean_costs.npy')
        eval_ep_arrays = load_seeds(args.results_dir, cfg['pattern'], 'eval_episodes.npy')

        mean_r, std_r = align_and_aggregate(reward_arrays)
        mean_c, std_c = align_and_aggregate(cost_arrays)
        mean_eval_r, std_eval_r = align_and_aggregate(eval_r_arrays)
        mean_eval_c, std_eval_c = align_and_aggregate(eval_c_arrays)

        results[method_name] = {
            'train_reward_mean': mean_r, 'train_reward_std': std_r,
            'train_cost_mean': mean_c, 'train_cost_std': std_c,
            'eval_reward_mean': mean_eval_r, 'eval_reward_std': std_eval_r,
            'eval_cost_mean': mean_eval_c, 'eval_cost_std': std_eval_c,
            'eval_episodes': eval_ep_arrays[0] if eval_ep_arrays else None,
            'n_seeds': len(reward_arrays),
        }
        if reward_arrays:
            print(f"[{method_name}] Loaded {len(reward_arrays)} seeds, {len(reward_arrays[0])} episodes each")
        else:
            print(f"[{method_name}] WARNING: No data found for pattern: {cfg['pattern']}")
    else:
        # No-action: load from JSON
        search = os.path.join(args.results_dir, cfg['pattern'])
        dirs = sorted(glob(search))
        rewards_all, costs_all = [], []
        for d in dirs:
            json_path = os.path.join(d, 'results.json')
            if os.path.exists(json_path):
                with open(json_path) as f:
                    r = json.load(f)
                rewards_all.append(r['mean_reward'])
                costs_all.append(r['mean_cost'])
        if rewards_all:
            results[method_name] = {
                'final_reward_mean': np.mean(rewards_all),
                'final_reward_std': np.std(rewards_all),
                'final_cost_mean': np.mean(costs_all),
                'final_cost_std': np.std(costs_all),
                'n_seeds': len(rewards_all),
            }
            print(f"[{method_name}] Loaded {len(rewards_all)} seeds")
        else:
            print(f"[{method_name}] WARNING: No data found for pattern: {cfg['pattern']}")


# ─── Plot 1: Training Reward Curves ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for method_name, cfg in METHODS.items():
    data = results.get(method_name, {})
    if not cfg['has_training_curve'] or data.get('train_reward_mean') is None:
        # Plot horizontal line for no-action
        if 'final_reward_mean' in data:
            for ax_idx in range(2):
                axes[ax_idx].axhline(y=data['final_reward_mean'] if ax_idx == 0 else data['final_cost_mean'],
                                     color=cfg['color'], linestyle=cfg['linestyle'],
                                     label=f"{method_name}", alpha=0.7)
        continue

    mean_r = smooth(data['train_reward_mean'], args.smooth_window)
    std_r = smooth(data['train_reward_std'], args.smooth_window)
    x = np.arange(len(mean_r))

    axes[0].plot(x, mean_r, color=cfg['color'], linestyle=cfg['linestyle'], label=method_name)
    axes[0].fill_between(x, mean_r - std_r, mean_r + std_r, color=cfg['color'], alpha=0.15)

    if data.get('train_cost_mean') is not None:
        mean_c = smooth(data['train_cost_mean'], args.smooth_window)
        std_c = smooth(data['train_cost_std'], args.smooth_window)
        x_c = np.arange(len(mean_c))
        axes[1].plot(x_c, mean_c, color=cfg['color'], linestyle=cfg['linestyle'], label=method_name)
        axes[1].fill_between(x_c, mean_c - std_c, mean_c + std_c, color=cfg['color'], alpha=0.15)

axes[0].set_xlabel('Episode')
axes[0].set_ylabel('Episode Reward')
axes[0].set_title('Training Reward')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('Episode')
axes[1].set_ylabel('Avg Step Cost')
axes[1].set_title('Training Cost (Headway Deviation)')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, 'training_curves.pdf'), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(args.output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
print(f"Saved training_curves.pdf/png")


# ─── Plot 2: Eval Reward & Cost Curves ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for method_name, cfg in METHODS.items():
    data = results.get(method_name, {})
    if not cfg['has_training_curve']:
        if 'final_reward_mean' in data:
            for ax_idx in range(2):
                axes[ax_idx].axhline(y=data['final_reward_mean'] if ax_idx == 0 else data['final_cost_mean'],
                                     color=cfg['color'], linestyle=cfg['linestyle'],
                                     label=f"{method_name}", alpha=0.7)
        continue

    if data.get('eval_reward_mean') is not None and data.get('eval_episodes') is not None:
        ep = data['eval_episodes'][:len(data['eval_reward_mean'])]
        axes[0].plot(ep, data['eval_reward_mean'], color=cfg['color'], linestyle=cfg['linestyle'],
                     label=method_name, marker='o', markersize=3)
        axes[0].fill_between(ep,
                             data['eval_reward_mean'] - data['eval_reward_std'],
                             data['eval_reward_mean'] + data['eval_reward_std'],
                             color=cfg['color'], alpha=0.15)

    if data.get('eval_cost_mean') is not None and data.get('eval_episodes') is not None:
        ep = data['eval_episodes'][:len(data['eval_cost_mean'])]
        axes[1].plot(ep, data['eval_cost_mean'], color=cfg['color'], linestyle=cfg['linestyle'],
                     label=method_name, marker='o', markersize=3)
        axes[1].fill_between(ep,
                             data['eval_cost_mean'] - data['eval_cost_std'],
                             data['eval_cost_mean'] + data['eval_cost_std'],
                             color=cfg['color'], alpha=0.15)

axes[0].set_xlabel('Episode')
axes[0].set_ylabel('Eval Mean Reward')
axes[0].set_title('Evaluation Reward (Deterministic Policy)')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('Episode')
axes[1].set_ylabel('Eval Avg Step Cost')
axes[1].set_title('Evaluation Cost (Headway Deviation)')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, 'eval_curves.pdf'), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(args.output_dir, 'eval_curves.png'), dpi=150, bbox_inches='tight')
print(f"Saved eval_curves.pdf/png")


# ─── Table: Final performance summary (LaTeX) ───────────────────────
print("\n" + "="*80)
print("FINAL PERFORMANCE SUMMARY")
print("="*80)
print(f"{'Method':<25} {'Reward (mean±std)':<25} {'Cost (mean±std)':<25} {'Seeds'}")
print("-"*80)

table_rows = []
for method_name in METHODS:
    data = results.get(method_name, {})
    if 'final_reward_mean' in data:
        r_mean, r_std = data['final_reward_mean'], data['final_reward_std']
        c_mean, c_std = data['final_cost_mean'], data['final_cost_std']
        n = data['n_seeds']
    elif data.get('eval_reward_mean') is not None and len(data['eval_reward_mean']) > 0:
        # Use last eval point
        r_mean = data['eval_reward_mean'][-1]
        r_std = data['eval_reward_std'][-1]
        c_mean = data['eval_cost_mean'][-1] if data.get('eval_cost_mean') is not None else float('nan')
        c_std = data['eval_cost_std'][-1] if data.get('eval_cost_std') is not None else float('nan')
        n = data['n_seeds']
    else:
        print(f"{method_name:<25} {'N/A':<25} {'N/A':<25} 0")
        continue

    print(f"{method_name:<25} {r_mean:>10.1f} ± {r_std:<10.1f} {c_mean:>10.2f} ± {c_std:<10.2f} {n}")
    table_rows.append((method_name, r_mean, r_std, c_mean, c_std))

# LaTeX table
latex_path = os.path.join(args.output_dir, 'results_table.tex')
with open(latex_path, 'w') as f:
    f.write("\\begin{table}[h]\n")
    f.write("\\centering\n")
    f.write("\\caption{Final evaluation performance (mean $\\pm$ std across 3 seeds)}\n")
    f.write("\\label{tab:results}\n")
    f.write("\\begin{tabular}{lcc}\n")
    f.write("\\toprule\n")
    f.write("Method & Reward ($\\uparrow$) & Cost ($\\downarrow$) \\\\\n")
    f.write("\\midrule\n")
    for name, r_m, r_s, c_m, c_s in table_rows:
        f.write(f"{name} & ${r_m:.0f} \\pm {r_s:.0f}$ & ${c_m:.1f} \\pm {c_s:.1f}$ \\\\\n")
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")

print(f"\nLaTeX table saved to {latex_path}")
print(f"All figures saved to {args.output_dir}/")
