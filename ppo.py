from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch.optim import Adam
from typing import Optional
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from pystk2_gymnasium import AgentSpec
from pystk2_gymnasium.stk_wrappers import ConstantSizedObservations
from stk_actor.core.wrappers import EnhanceObservations, CustomPolarObservations, StuckObservations, ExpertObservations
from stk_actor.core.nn import ActorMLP, StochasticActorMLP, CriticMLP
from stk_actor.core.state import State
from stk_actor.core.action import Action
import neptune
import os
from dotenv import load_dotenv
from tqdm import tqdm
import argparse


@dataclass
class RolloutBatch:
    states: torch.Tensor         # [B, obs_dim]
    actions: list[torch.Tensor]  # list of [B]
    old_logp: torch.Tensor       # [B]
    returns: torch.Tensor        # [B]
    advantages: torch.Tensor     # [B]


class RolloutBuffer:
    def __init__(self, buffer_size: int, obs_dim: int, n_actions: int, device: str):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.device = device
        self.reset()

    def reset(self):
        self.states = []
        self.actions = [[] for _ in range(self.n_actions)]
        self.rewards = []
        self.terminated = []   # 1.0 if terminated, else 0.0 (bootstrap on truncation)
        self.values = []
        self.logp = []

        self.advantages = None
        self.returns = None

    def add(self, state, actions_list, reward, terminated, value, logp):
        self.states.append(state)
        for i in range(self.n_actions):
            self.actions[i].append(actions_list[i])
        self.rewards.append(reward)
        self.terminated.append(terminated)
        self.values.append(value)
        self.logp.append(logp)

    def compute_returns_and_advantages(self, last_value, gamma=0.99, gae_lambda=0.95):
        """Computes GAE advantages and returns.

        Args:
            last_value: value of the last state
            gamma: discount factor
            gae_lambda: GAE lambda parameter
        """
        rewards = torch.stack(self.rewards).to(self.device)               # [T]
        terminated = torch.stack(self.terminated).to(self.device)         # [T]
        values = torch.stack(self.values).to(self.device).squeeze(-1)     # [T]
        # logp = torch.stack(self.logp).to(self.device)                     # [T]

        T = rewards.shape[0]
        advantages = torch.zeros(T, device=self.device)
        last_gae = 0.0

        # next_values[t] = values[t+1], and for t=T-1 use last_value
        next_values = torch.zeros_like(values)
        next_values[:-1] = values[1:]
        next_values[-1] = last_value

        # must_bootstrap = 1 - terminated
        must_bootstrap = 1.0 - terminated

        deltas = rewards + gamma * next_values * must_bootstrap - values

        for t in reversed(range(T)):
            last_gae = deltas[t] + gamma * gae_lambda * must_bootstrap[t] * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        self.advantages = advantages
        self.returns = returns

    def get_batches(self, batch_size: int, shuffle=True):
        states = torch.stack(self.states).to(self.device)  # [T, obs_dim]
        actions = [torch.stack(a).to(self.device) for a in self.actions]  # list of [T]
        old_logp = torch.stack(self.logp).to(self.device)  # [T]
        returns = self.returns.to(self.device)             # [T]
        adv = self.advantages.to(self.device)              # [T]

        # advantage normalization (SB3 default)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        T = states.shape[0]
        idx = torch.randperm(T) if shuffle else torch.arange(T)

        for start in range(0, T, batch_size):
            mb_idx = idx[start:start + batch_size]
            yield RolloutBatch(
                states=states[mb_idx],
                actions=[a[mb_idx] for a in actions],
                old_logp=old_logp[mb_idx],
                returns=returns[mb_idx],
                advantages=adv[mb_idx],
            )


class PPOTrainer:
    def __init__(
        self,
        pretrained_policy_path: Optional[str] = None,
        pretrained_critic_path: Optional[str] = None,
        self_play: bool = False,
        n_steps: int = 2048,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        lr: float = 1e-3,
        n_epochs: int = 4,
        batch_size: int = 256,
        all_features: bool = False,
        logging: bool = False,
        experiment_name: Optional[str] = None,
        tags: list[str] = [],
    ):
        # Load environment variables
        load_dotenv(override=True)
        # Setup neptune for logging if enabled
        self.logging = logging
        if self.logging:
            neptune_project = os.getenv("NEPTUNE_PROJECT")
            neptune_api_token = os.getenv("NEPTUNE_API_TOKEN")
            if neptune_project is None:
                raise ValueError("NEPTUNE_PROJECT environment variable not set.")
            if neptune_api_token is None:
                raise ValueError("NEPTUNE_API_TOKEN environment variable not set.")
            self.neptune_run = neptune.init_run(
                project=neptune_project,
                api_token=neptune_api_token,
                name=experiment_name,
                tags=tags
            )
            params = {
                "pretrained_policy": pretrained_policy_path is not None,
                "pretrained_critic": pretrained_critic_path is not None,
                "self_play": self_play,
                "n_steps": n_steps,
                "gamma": gamma,
                "gae_lambda": gae_lambda,
                "clip_range": clip_range,
                "ent_coef": ent_coef,
                "vf_coef": vf_coef,
                "max_grad_norm": max_grad_norm,
                "lr": lr,
                "n_epochs": n_epochs,
                "batch_size": batch_size,
                "hidden_dim": 512
            }
            self.neptune_run["parameters"] = params
            # Upload a copy of the state representation source code
            self.neptune_run['src/state.py'].upload('stk_actor/core/state.py')
            # Upload a copy of the action representation source code
            self.neptune_run['src/action.py'].upload('stk_actor/core/action.py')
        # Set self-play flag
        self.self_play = self_play
        # Initialize the environment with necessary wrappers
        self.env = gym.make("supertuxkart/full-v0", render_mode=None, agent=AgentSpec(use_ai=False), difficulty=2, num_kart=10)
        wrappers = [
            lambda env: TimeLimit(env, max_episode_steps=2500),
            lambda env: ConstantSizedObservations(env, state_paths=10),
            lambda env: EnhanceObservations(env),
            lambda env: StuckObservations(env),
            lambda env: ExpertObservations(env),
            lambda env: CustomPolarObservations(env)
        ]
        for wrapper in wrappers:
            self.env = wrapper(self.env)
        # Check device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        # Set all features flag
        self.all_features = all_features
        # Get observation and action dimensions
        state, _ = self.env.reset()
        self.obs_dim = State.to_model(State.from_env(state), all_features=self.all_features).shape[0]
        self.n_actions = len(Action.get_types())
        # Initialize models
        # Policy model
        pretrained_policy = None
        if pretrained_policy_path is not None:
            pretrained_policy = ActorMLP(input_dim=self.obs_dim, hidden_dim=512).to(self.device)
            pretrained_state = torch.load(pretrained_policy_path, weights_only=True, map_location=self.device)
            pretrained_policy.load_state_dict(pretrained_state)
            print(f"Loaded pretrained policy from {pretrained_policy_path}")
        self.actor = StochasticActorMLP(
            input_dim=self.obs_dim,
            hidden_dim=512,
            pretrained_mlp=pretrained_policy,
        ).to(self.device)
        # Critic model
        self.critic = CriticMLP(
            input_dim=self.obs_dim,
            hidden_dim=512,
        ).to(self.device)
        if pretrained_critic_path is not None:
            pretrained_critic_state = torch.load(pretrained_critic_path, weights_only=True, map_location=self.device)
            self.critic.load_state_dict(pretrained_critic_state)
            print(f"Loaded pretrained critic from {pretrained_critic_path}")
        # PPO Hyperparameters
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        # Training parameters
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        # Optimizers
        self.actor_opt = Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = Adam(self.critic.parameters(), lr=lr)
        # Rollout buffer
        self.buffer = RolloutBuffer(n_steps, self.obs_dim, self.n_actions, self.device)

    def _env_obs_to_tensor(self, obs):
        # Your project-specific conversion
        s = State.to_model(State.from_env(obs, all_features=self.all_features))
        return s.to(self.device)

    def _actions_list_to_env(self, actions_list):
        # Convert list of tensors (scalars) to your env Action
        # action_pred is typically concatenated in your code. We'll mimic that.
        a_vec = torch.stack(actions_list, dim=-1)  # [n_actions] for single step
        return Action.to_env(Action.from_model(a_vec.detach().cpu()))

    def collect_rollouts(self):
        self.buffer.reset()

        obs, _ = self.env.reset()
        state = self._env_obs_to_tensor(obs)

        # Track if episode continues
        for step in range(self.n_steps):
            with torch.no_grad():
                # actor expects batch, so unsqueeze
                actions_list, logp, ent = self.actor.sample(state.unsqueeze(0))
                # list of [1] -> scalar
                actions_list = [a.squeeze(0) for a in actions_list]
                logp = logp.squeeze(0)

                value = self.critic(state.unsqueeze(0)).squeeze(0)  # scalar tensor

            env_action = self._actions_list_to_env(actions_list)
            next_obs, reward, terminated, truncated, info = self.env.step(env_action)

            terminated_flag = float(terminated)  # terminated stops bootstrap; truncated does NOT

            self.buffer.add(
                state,
                actions_list,
                torch.tensor(float(reward), device=self.device),
                torch.tensor(terminated_flag, device=self.device),
                value,
                logp,
            )

            if terminated or truncated:
                next_obs, _ = self.env.reset()
                if self.logging:
                    self.neptune_run['train/reward'].log(reward)

            state = self._env_obs_to_tensor(next_obs)

        # Bootstrap value for last state
        with torch.no_grad():
            last_value = self.critic(state.unsqueeze(0)).squeeze(0)

        self.buffer.compute_returns_and_advantages(
            last_value=last_value,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

    def update(self):
        # Prepare for logging
        policy_losses = []
        value_losses = []
        total_losses = []
        # PPO update epochs
        for epoch in range(self.n_epochs):
            for batch in self.buffer.get_batches(self.batch_size, shuffle=True):
                # Evaluate current policy on batch actions
                _, logp_new, entropy = self.actor.sample(batch.states, actions=batch.actions)
                values = self.critic(batch.states).squeeze(-1)

                # PPO ratio
                ratio = torch.exp(logp_new - batch.old_logp)

                # clipped surrogate
                surr1 = ratio * batch.advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * batch.advantages
                policy_loss = -torch.mean(torch.min(surr1, surr2))
                policy_losses.append(policy_loss.item())

                # value loss
                value_loss = F.mse_loss(values, batch.returns)
                value_losses.append(value_loss.item())

                # entropy bonus (maximize entropy -> subtract in loss)
                ent_loss = -torch.mean(entropy)

                total_loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * ent_loss
                total_losses.append(total_loss.item())

                # Optimize actor+critic together or separately
                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

                self.actor_opt.step()
                self.critic_opt.step()
        # Logging losses
        if self.logging:
            self.neptune_run["train/policy_loss"].log(sum(policy_losses) / len(policy_losses))
            self.neptune_run["train/value_loss"].log(sum(value_losses) / len(value_losses))
            self.neptune_run["train/total_loss"].log(sum(total_losses) / len(total_losses))

    def train(self, total_timesteps: int = 100000, log_interval: int = 1,
              model_save_interval: int = 10, policy_save_path: str = "ppo_policy.pth",
              stochastic_policy_save_path: str = "ppo_stochastic_policy.pth", critic_save_path: str = "ppo_critic.pth"):
        n_updates = total_timesteps // self.n_steps
        tqdm_bar = tqdm(range(1, n_updates + 1), desc=f"Iteration 0/{n_updates}:")
        for update in tqdm_bar:
            # PPO iteration
            self.collect_rollouts()
            self.update()
            # Logging stats
            if update % log_interval == 0:
                # simple logging
                avg_return = self.buffer.returns.mean().item()
                avg_adv = self.buffer.advantages.mean().item()
                if self.logging:
                    self.neptune_run["train/avg_return"].log(avg_return)
                    self.neptune_run["train/avg_advantage"].log(avg_adv)
            # Model saving
            if update % model_save_interval == 0:
                # Save models
                torch.save(self.actor.deterministic_actor.state_dict(), policy_save_path)
                torch.save(self.critic.state_dict(), critic_save_path)
                # Also save stochastic policy
                torch.save(self.actor.state_dict(), stochastic_policy_save_path)
                if self.logging:
                    self.neptune_run["models/policy"].upload(policy_save_path)
                    self.neptune_run["models/critic"].upload(critic_save_path)
                    self.neptune_run["models/stochastic_policy"].upload(stochastic_policy_save_path)
            # Update tqdm description
            tqdm_bar.set_description(f"Iteration {update}/{n_updates}:")
        # Save final models
        torch.save(self.actor.deterministic_actor.state_dict(), policy_save_path)
        torch.save(self.critic.state_dict(), critic_save_path)
        # Also save stochastic policy
        torch.save(self.actor.state_dict(), stochastic_policy_save_path)
        if self.logging:
            self.neptune_run["models/policy"].upload(policy_save_path)
            self.neptune_run["models/critic"].upload(critic_save_path)
            self.neptune_run["models/stochastic_policy"].upload(stochastic_policy_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent.")
    parser.add_argument("--pretrained-policy-path", type=str, default=None, help="Path to pretrained policy model.")
    parser.add_argument("--pretrained-critic-path", type=str, default=None, help="Path to pretrained critic model.")
    parser.add_argument("--self-play", action="store_true", help="Enable self-play training.")
    parser.add_argument("--n-steps", type=int, default=2048, help="Number of steps per rollout.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda parameter.")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range.")
    parser.add_argument("--ent-coef", type=float, default=0.0, help="Entropy coefficient.")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value function coefficient.")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Max gradient norm.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--n-epochs", type=int, default=4, help="Number of epochs per update.")
    parser.add_argument("--batch-size", type=int, default=256, help="Minibatch size.")
    parser.add_argument("--all-features", action="store_true", help="Use all features in state representation.")
    parser.add_argument("--logging", "-l", action="store_true", help="Enable Neptune logging.")
    parser.add_argument("--experiment-name", type=str, default=None, help="Neptune experiment name.")
    parser.add_argument("--tags", type=str, nargs='*', default=[], help="Neptune tags.")
    parser.add_argument("--total-timesteps", type=int, default=100000, help="Total training timesteps.")
    parser.add_argument("--log-interval", type=int, default=1, help="Logging interval (in updates).")
    parser.add_argument("--model-save-interval", type=int, default=10, help="Model saving interval (in updates).")
    parser.add_argument("--policy-save-path", type=str, default="ppo_policy.pth", help="Path to save policy model.")
    parser.add_argument("--stochastic-policy-save-path", type=str, default="ppo_stochastic_policy.pth", help="Path to save stochastic policy model.")
    parser.add_argument("--critic-save-path", type=str, default="ppo_critic.pth", help="Path to save critic model.")

    args = parser.parse_args()

    trainer = PPOTrainer(
        pretrained_policy_path=args.pretrained_policy_path,
        pretrained_critic_path=args.pretrained_critic_path,
        self_play=args.self_play,
        n_steps=args.n_steps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        lr=args.lr,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        all_features=args.all_features,
        logging=args.logging,
        experiment_name=args.experiment_name,
        tags=args.tags
    )
    trainer.train(
        total_timesteps=args.total_timesteps,
        log_interval=args.log_interval,
        model_save_interval=args.model_save_interval,
        policy_save_path=args.policy_save_path,
        stochastic_policy_save_path=args.stochastic_policy_save_path,
        critic_save_path=args.critic_save_path
    )
