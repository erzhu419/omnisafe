"""
No-action baseline: all buses hold for 0 seconds at every station.
This serves as the lower-bound baseline for comparison.
"""
import os
import argparse
import numpy as np
import json
from env.sim import env_bus

parser = argparse.ArgumentParser(description='No-action baseline evaluation')
parser.add_argument('--num_episodes', type=int, default=30, help='Number of evaluation episodes')
parser.add_argument('--route_sigma', type=float, default=1.5, help='Sigma for route speed sampling')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--save_root', type=str, default='results_no_action', help='Output directory')
args = parser.parse_args()

np.random.seed(args.seed)

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'env')
env = env_bus(path, debug=False, route_sigma=args.route_sigma)

sigma_token = f"sigma{args.route_sigma}".replace('.', 'p')
save_dir = os.path.join(args.save_root, f"no_action_{sigma_token}_seed{args.seed}")
os.makedirs(save_dir, exist_ok=True)

all_rewards = []
all_costs = []

for ep in range(args.num_episodes):
    env.reset()
    state_dict, reward_dict, cost_dict, _ = env.initialize_state(render=False)

    done = False
    episode_reward = 0
    episode_cost = 0
    episode_steps = 0
    action_dict = {key: 0 for key in range(env.max_agent_num)}

    while not done:
        for key in state_dict:
            if len(state_dict[key]) == 2:
                if state_dict[key][0][1] != state_dict[key][1][1]:
                    episode_reward += reward_dict[key]
                    episode_cost += cost_dict[key]
                    episode_steps += 1
                state_dict[key] = state_dict[key][1:]
            action_dict[key] = 0

        state_dict, reward_dict, cost_dict, done = env.step(action_dict, render=False)

    avg_cost = episode_cost / max(1, episode_steps)
    all_rewards.append(episode_reward)
    all_costs.append(avg_cost)
    print(f"Episode {ep}: reward={episode_reward:.2f}, avg_cost={avg_cost:.2f}")

results = {
    "algorithm": "no_action",
    "route_sigma": args.route_sigma,
    "seed": args.seed,
    "num_episodes": args.num_episodes,
    "mean_reward": float(np.mean(all_rewards)),
    "reward_std": float(np.std(all_rewards)),
    "mean_cost": float(np.mean(all_costs)),
    "cost_std": float(np.std(all_costs)),
}

np.save(os.path.join(save_dir, 'rewards.npy'), all_rewards)
np.save(os.path.join(save_dir, 'costs.npy'), all_costs)
with open(os.path.join(save_dir, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n=== No-Action Baseline ===")
print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['reward_std']:.2f}")
print(f"Mean Cost:   {results['mean_cost']:.2f} ± {results['cost_std']:.2f}")
print(f"Results saved to {save_dir}")
