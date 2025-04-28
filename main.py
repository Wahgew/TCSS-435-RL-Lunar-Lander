# main.py
import os
import torch
import gymnasium as gym
import argparse
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
from typing import Dict, Any

from agents.dqn_agent import DQNAgent
from utils.train_logger import TrainLogger


def train(
        env: gym.Env,
        agent: DQNAgent,
        n_episodes: int = 2000,  # Increased from 1000
        max_t: int = 1000,
        output_dir: str = "results",
) -> Dict[str, Any]:
    """
    Train the DQN agent.

    Args:
        env: Environment
        agent: Agent to train
        n_episodes: Maximum number of training episodes (increased for better learning)
        max_t: Maximum number of timesteps per episode
        output_dir: Directory to save outputs

    Returns:
        Dictionary with training metrics
    """
    logger = TrainLogger()
    episode_rewards = []

    # Record start time
    start_time = time.time()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print("\nTraining DQN agent...")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    for i_episode in range(1, n_episodes + 1):
        # Reset environment
        state, _ = env.reset()
        total_reward = 0
        returns = 0
        success = False
        gamma_t = 1.0  # For calculating return (discounted reward)

        # Episode loop
        for t in range(max_t):
            # Select and perform an action
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Process step
            agent.step(state, action, reward, next_state, done)

            # Update metrics
            state = next_state
            total_reward += reward
            returns += gamma_t * reward
            gamma_t *= agent.gamma

            if done:
                # Check success (landing between flags with low velocity)
                success = terminated and not truncated and total_reward > 100
                break

        # Update epsilon after each episode
        agent.update_epsilon()

        # Log episode metrics
        episode_rewards.append(total_reward)
        metrics = logger.log_episode(i_episode, total_reward, returns, success)

        # Print progress
        if i_episode % 10 == 0:
            elapsed = time.time() - start_time
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            print(f"Episode {i_episode}/{n_episodes} | Score: {total_reward:.2f} | "
                  f"Avg Score: {metrics['avg_score']:.2f} | Success Rate: {metrics['success_rate']:.2f}% | "
                  f"Epsilon: {agent.eps_start:.4f} | Elapsed: {elapsed_str}")

        # Check if environment is solved
        if metrics['avg_score'] >= 200.0 and i_episode >= 100:
            elapsed = time.time() - start_time
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            print(f"\nEnvironment solved in {i_episode:d} episodes! "
                  f"Average Score: {metrics['avg_score']:.2f} | Total time: {elapsed_str}")
            agent.save(os.path.join(run_dir, "solved_model.pth"))
            break

    # Calculate total training time
    total_time = time.time() - start_time
    total_time_str = str(timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")

    # Save final model and training logs
    agent.save(os.path.join(run_dir, "final_model.pth"))
    logger.save_to_csv(os.path.join(run_dir, "training_log.csv"))
    logger.plot_training(os.path.join(run_dir, "training_plot.png"))

    return {
        "episodes": i_episode,
        "final_avg_score": metrics['avg_score'],
        "final_success_rate": metrics['success_rate'],
        "run_dir": run_dir,
        "training_time": total_time_str
    }


def main() -> None:
    """Main function to set up and run DQN training."""
    parser = argparse.ArgumentParser(description="Train a DQN agent on LunarLander-v3")
    parser.add_argument("--episodes", type=int, default=2000, help="Number of episodes to train")
    parser.add_argument("--buffer_size", type=int, default=100000, help="Replay buffer size")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=1e-3, help="Soft update parameter")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--update_every", type=int, default=4, help="Update frequency")
    parser.add_argument("--eps_start", type=float, default=1.0, help="Starting epsilon")
    parser.add_argument("--eps_end", type=float, default=0.01, help="Minimum epsilon")
    parser.add_argument("--eps_decay", type=float, default=0.995, help="Epsilon decay factor")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    # Create environment
    env = gym.make("LunarLander-v3", render_mode=None)  # No rendering during training
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Set up device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create agent
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        device=device,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        lr=args.lr,
        update_every=args.update_every,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay=args.eps_decay
    )

    # Train agent
    results = train(
        env=env,
        agent=agent,
        n_episodes=args.episodes,
        output_dir=args.output_dir,
    )

    print(f"\nTraining complete!")
    print(f"Results saved to: {results['run_dir']}")
    print(f"Final average score: {results['final_avg_score']:.2f}")
    print(f"Final success rate: {results['final_success_rate']:.2f}%")
    print(f"Total training time: {results['training_time']}")

    # Close environment
    env.close()

    # Try to display the final plots
    try:
        plot_path = os.path.join(results['run_dir'], "training_plot.png")
        if os.path.exists(plot_path):
            img = plt.imread(plot_path)
            plt.figure(figsize=(10, 15))
            plt.imshow(img)
            plt.axis('off')
            plt.show()
    except Exception as e:
        print(f"Could not display plot: {e}")


if __name__ == "__main__":
    main()