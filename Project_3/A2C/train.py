import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import imageio.v2 as imageio

from actor import Actor
from critic import Critic
from compute_objectives import *

from utils import *


def moving_average(values, window_size):
    if window_size <= 1 or len(values) == 0:
        return values
    window_size = min(window_size, len(values))
    kernel = np.ones(window_size) / window_size
    return np.convolve(values, kernel, mode="valid")

def normalize_observation(observation, obs_normalizer=None):
    if obs_normalizer is None:
        return observation
    return obs_normalizer.normalize(observation)


def run_lunar_lander(actor=None, video_filename="lunar_lander_example.mp4", config=None):
    """Run the actor on LunarLander and save a video of the run."""
    config = config or load_config()
    env = make_env(config["env_id"])
    state = reset_env(env)
    obs_normalizer = getattr(actor, "obs_normalizer", None) if actor is not None else None
    total_reward = 0.0
    frames = []

    for _ in range(config["max_ep_steps"]):
        frame = render_frame(env)
        frame_array = np.require(np.asarray(frame, dtype=np.uint8), requirements=["C", "A", "O"])
        if frame_array.ndim in (2, 3):
            frames.append(frame_array)

        if actor is None:
            action = env.action_space.sample()
        else:
            normalized_state = normalize_observation(state, obs_normalizer)
            state_tensor = torch.tensor(normalized_state, dtype=torch.float32)
            action = actor.get_action(state_tensor, deterministic=True)
        # print(f"action: {action}")
        

        state, reward, terminated, truncated, _ = step_env(env, action)
        total_reward += reward

        if terminated or truncated:
            print("Reward: ", str(total_reward))
            break

    env.close()

    if not frames:
        print(f"Skipping video generation for {video_filename}: no frames were rendered.")
        return

    if not os.path.exists(VIDEOS_DIR):
        os.makedirs(VIDEOS_DIR)
    video_path = os.path.join(VIDEOS_DIR, video_filename)
    imageio.mimsave(video_path, frames, fps=20, macro_block_size=1)

def test_actor(actor, env, obs_normalizer, i_episode):
    # Test loss every 10 episodes
    raw_state = reset_env(env, seed=config["random_seed"] if i_episode == 0 else None)
    obs_normalizer.update(raw_state)
    state = torch.tensor(normalize_observation(raw_state, obs_normalizer), dtype=torch.float32)
    episode_states = []
    episode_actions = []
    episode_rewards = []
    episode_terminated = False
    episode_truncated = False
    state_batch: torch.tensor = state
    action_batch = torch.tensor([0])
    if i_episode % 10 == 0:
        with torch.no_grad():
            for _ in range(config["max_ep_steps"]):
                if episode_terminated or episode_truncated:
                    break
                # Interact with the environment for one step and record the transition
                # Hint: This block should choose an action from the actor, step the environment,
                # update the observation statistics, and save the information needed later
                # to build returns and losses.

                # Take action and update env
                normalized_state = normalize_observation(state, obs_normalizer)
                state_tensor = torch.tensor(normalized_state, dtype=torch.float32)
                # print("getting action")
                action = actor.get_action(state_tensor, True)
                # print(f"action {action} acquired")
                # action = actor.get_action(state, deterministic=True)
                next_state, reward, episode_terminated, episode_truncated, info = step_env(env, action)
                obs_normalizer.update(next_state)
                next_state = torch.tensor(normalize_observation(next_state, obs_normalizer), dtype=torch.float32)
                
                # Store data
                episode_rewards.append(reward)
                episode_states.append(next_state)
                episode_actions.append(action)
                state_batch = torch.vstack([state_batch, next_state])
                action_batch = torch.vstack([action_batch, torch.tensor(action)])

                state = next_state
            # Convert the collected episode data into batched tensors
            # print(state_batch)
            state_batch = state_batch[1:]
            action_batch = action_batch[1:]
            # reward_history[i_episode] = np.sum(episode_rewards) # not discounted sum

            # assert len(state_batch) == len(action_batch), f"Values should be equal. |state_batch_len = {len(state_batch)}| |action_batch_len = {len(action_batch)}|"
            # print(f"Episode rewards: {episode_rewards}")
            # Hint: Use the stored rewards together with gamma 
            return_batch = compute_discounted_returns(episode_rewards, gamma=config["gamma"]) 

            # Evaluate the log-probabilities of the actions that were actually taken
            chosen_log_probs, entropy = actor.evaluate_actions(state_batch, action_batch)
            # print(f"Training episode {i_episode}/{config['num_episodes']}") # Current Entropy = {entropy}")

            # Policy gradient update
            actor_loss = compute_actor_loss(chosen_log_probs, return_batch, config['grad_norm_clip'])
            actor_loss.requires_grad = True

            print(f"actor_loss for episode {i_episode} in testing: {actor_loss}")
            print(f"Total reward: {np.sum(episode_rewards)}")
            print(f"Entropy: {entropy.mean()}")
    
    reward_history = np.sum(episode_rewards)

    return reward_history


def train_actor_critic(config_path=None, plot=True):
    config = load_config(config_path) if config_path else load_config()
    set_random_seed(config["random_seed"])

    env = make_env(config["env_id"])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    algorithm = config["algorithm"].lower()
    #ensure that the algorithm is either "a2c" or "reinforce"
    if algorithm not in ["a2c", "reinforce"]:
        raise ValueError(f"Invalid algorithm: {algorithm}. Must be 'a2c' or 'reinforce'.")

    # if the algorithm is A2C we use the critic, if not we just use the actor (REINFORCE)
    use_a2c = algorithm == "a2c"
    use_reinforce = algorithm == "reinforce"

    print("state dim: ", state_dim)
    print("action dim: ", action_dim)
    print("algorithm: ", algorithm)

    actor = Actor(state_dim, action_dim, config["hidden_dim"])
    critic = Critic(state_dim, config["hidden_dim"]) if use_a2c else None
    #This helps training
    obs_normalizer = ObservationNormalizer(state_dim)
    actor.obs_normalizer = obs_normalizer

    actor_optim = optim.Adam(actor.parameters(), lr=config["actor_lr"])
    actor_optim.zero_grad()
    if use_a2c:
        critic_optim = (optim.Adam(critic.parameters(), lr=config["critic_lr"]) if use_a2c else None)
        critic_optim.zero_grad()

    reward_history = np.zeros(config["num_episodes"])/10

    for i_episode in range(config["num_episodes"]):
        raw_state = reset_env(env, seed=config["random_seed"] if i_episode == 0 else None)
        obs_normalizer.update(raw_state)
        state = torch.tensor(normalize_observation(raw_state, obs_normalizer), dtype=torch.float32)
        episode_reward = 0.0
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_ends = []
        episode_terminated = False
        episode_truncated = False
        state_batch: torch.tensor = state
        action_batch = torch.tensor([0])

        # Begin iterating through time 
        # This is to prevent pathological cases where the episode never ends, we limit the number of steps per episode to max_ep_steps, but in practice for lunar lander it should end well before that
        for _ in range(config["max_ep_steps"]):
            if episode_terminated or episode_truncated:
                break
            # TODO: Interact with the environment for one step and record the transition
            # Hint: This block should choose an action from the actor, step the environment,
            # update the observation statistics, and save the information needed later
            # to build returns and losses.

            # Take action and update env
            action = actor.get_action(state, False)
            next_state, reward, episode_terminated, episode_truncated, info = step_env(env, action)
            obs_normalizer.update(next_state)
            next_state = torch.tensor(normalize_observation(next_state, obs_normalizer), dtype=torch.float32)
            done = int(episode_terminated | episode_truncated)

            # Store data
            episode_rewards.append(reward)
            episode_states.append(next_state)
            episode_actions.append(action)
            episode_reward += reward
            state_batch = torch.vstack([state_batch, next_state])
            action_batch = torch.vstack([action_batch, torch.tensor(action)])
            episode_ends.append(done)

            state = next_state

        # Convert the collected episode data into batched tensors
        # print(state_batch)
        state_batch = state_batch[1:]
        action_batch = action_batch[1:]
        # reward_history[i_episode] = episode_reward
        # reward_history[i_episode] = np.sum(episode_rewards) # not discounted sum

        

        # assert len(state_batch) == len(action_batch), f"Values should be equal. |state_batch_len = {len(state_batch)}| |action_batch_len = {len(action_batch)}|"
        # print(f"Episode rewards: {episode_rewards}")
        # Hint: Use the stored rewards together with gamma 
        return_batch = compute_discounted_returns(episode_rewards, gamma=config["gamma"]) 

        # Evaluate the log-probabilities of the actions that were actually taken
        chosen_log_probs, entropy = actor.evaluate_actions(state_batch, action_batch)
        # print(f"Training episode {i_episode}/{config['num_episodes']}") # Current Entropy = {entropy}")

        # if i_episode % 10 == 0:
            # print(f"Stochastic Entropy: {entropy}")


        # print(f"actor_loss: {actor_loss}")

        # optional 
        # actor_loss -=  config['value_loss_coef'] * entropy

        if use_reinforce: #this is the REINFORCE case where we don't use a critic, so the advantage is just the return
            # print("Using REINFORCE")
            # Policy-gradient update for REINFORCE
            # Policy gradient update
            actor_loss = compute_actor_loss(chosen_log_probs, return_batch, config['grad_norm_clip'])
            actor_loss.requires_grad = True
            # Backpropagation & optimization
            # Clear any stale actor gradients before backpropagation
            # Hint: Optimizers in PyTorch accumulate gradients unless you reset them.
            # with torch.no_grad(): 
            actor_loss.backward()
            actor_optim.step()
            actor_optim.zero_grad()


        elif use_a2c:#This is the critic case where we compute the advantage using the critic's value estimates, and use that to compute the actor loss, and also compute the critic loss and backprop through both
            # print("Using A2C")
            # actor-critic update
            # Hint: This branch should involve the critic's value estimates, an advantage term,
            # and a combined loss that updates both networks.
            value_batch = critic(state_batch)
            
            # actor_loss.requires_grad = True

            next_target = torch.tensor(episode_rewards) + (1 - torch.tensor(episode_ends)) * config["gamma"] * value_batch

            critic_loss = compute_critic_loss(next_target, value_batch, config['value_loss_coef'])
            # critic_loss.requires_grad = True
            advantage = compute_advantage(return_batch, value_batch)
            actor_loss = compute_actor_loss(chosen_log_probs, advantage, config['grad_norm_clip'])
            
            # Perform backprop
            with torch.no_grad(): 
                actor_loss.backward(retain_graph=True)
                critic_loss.backward(retain_graph=True)
            actor_optim.step()
            actor_optim.zero_grad()
            critic_optim.step()
            critic_optim.zero_grad()

            # print(f"critic_loss: {critic_loss}")


        reward_history[i_episode] = test_actor(actor, env, obs_normalizer, i_episode)
    print(f"avg reward: {reward_history.mean()}")

                            


    if plot:
        plt.figure()
        plt.plot(reward_history, label="Reward", alpha=0.35)
        ma_window = config["reward_ma_window"]
        reward_ma = moving_average(reward_history, ma_window)
        ma_x = np.arange(len(reward_ma)) + ma_window - 1
        plt.plot(ma_x, reward_ma, label=f"Reward MA ({ma_window})")
        plt.xlabel("Number of Episodes")
        plt.ylabel("Episode Reward")
        plt.title("History of Episode Reward")
        plt.legend()
        if not os.path.exists(PLOTS_DIR):
            os.makedirs(PLOTS_DIR)

        filename = f"{config['env_id']}_{config['algorithm']}_{config['random_seed']}_{config['plot_filename']}"
        plt.savefig(os.path.join(PLOTS_DIR, filename))
        plt.show()

    checkpoint_path = config["checkpoint_path"]
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "obs_normalizer_state": obs_normalizer.state_dict(),
            "state_dim": state_dim,
            "action_dim": action_dim,
            "hidden_dim": config["hidden_dim"],
            "config": config,
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")

    env.close()
    return actor


if __name__ == "__main__":
    config = load_config()
    #run_lunar_lander(None, "random_lunar_lander_example1.mp4", config=config)
    actor = train_actor_critic(plot=True)
    run_lunar_lander(actor, "trained_lunar_lander_example1.mp4", config=config)
