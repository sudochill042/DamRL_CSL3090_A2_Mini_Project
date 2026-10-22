"""Train a DQN or Double DQN agent.

Examples
  python scripts/train_dqn.py --algo dqn  --scenario ordinary --episodes 300 --seed 0
  python scripts/train_dqn.py --algo ddqn --scenario mixed    --episodes 300 --seed 0 --mask

Outputs
  models/<tag>_seed<seed>.pt          best-validation checkpoint
  results/train_<tag>_seed<seed>.csv  per-episode log
where tag = <algo>[_mask]_<scenario>.
"""
import argparse

from damrl.agent import DQNAgent, DQNConfig
from damrl.config import load_config
from damrl.env import ReservoirEnv
from damrl.evaluation import build_episode_set
from damrl.train import SCENARIO_MIXES, train


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="dqn", choices=["dqn", "ddqn"])
    p.add_argument("--scenario", default="ordinary", choices=list(SCENARIO_MIXES))
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mask", action="store_true", help="use action masking")
    p.add_argument("--no-forecast", action="store_true", help="drop the inflow forecast from the state")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--val-every", type=int, default=25)
    p.add_argument("--tag", default=None, help="override the output name")
    args = p.parse_args()

    cfg = load_config()
    env = ReservoirEnv(cfg, scenario=SCENARIO_MIXES[args.scenario],
                       use_forecast=not args.no_forecast, seed=args.seed)
    agent_cfg = DQNConfig(double=(args.algo == "ddqn"), use_mask=args.mask,
                          lr=args.lr, gamma=args.gamma)
    agent = DQNAgent(env.obs_dim, env.n_actions, agent_cfg, seed=args.seed)

    tag = args.tag or f"{args.algo}{'_mask' if args.mask else ''}" \
                      f"{'_nofc' if args.no_forecast else ''}_{args.scenario}"
    val = build_episode_set(cfg, "val")
    print(f"training {tag} seed {args.seed}: {args.episodes} episodes, obs_dim {env.obs_dim}")
    log = train(env, agent, args.episodes, seed=args.seed, val_episodes=val,
                val_every=args.val_every,
                log_path=f"results/train_{tag}_seed{args.seed}.csv",
                ckpt_path=f"models/{tag}_seed{args.seed}.pt")
    best = log["val_return"].max()
    print(f"done. best validation return {best:.1f}; checkpoint models/{tag}_seed{args.seed}.pt")


if __name__ == "__main__":
    main()
