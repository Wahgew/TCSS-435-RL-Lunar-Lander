import os
import torch
import gymnasium as gym
import argparse
import matplotlib.pyplot as plt
import time
import numpy as np
import random
from typing import Dict, Any
from agents.dqn_agent import DQNAgent
from agents.double_dqn_agent import DoubleDQNAgent  
from utils.train_logger import TrainLogger
from datetime import datetime, timedelta


def train(
        env: gym.Env,
        agent: DQNAgent,
        n_episodes: int = 2000,  # Increased from 1000
        max_t: int = 1000,
        output_dir: str = "results",
        agent_type: str = "DQN"
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
    run_dir = os.path.join(output_dir, f"{agent_type}_{timestamp}")
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
        "final_avg_return": metrics["avg_return"],
        "final_success_rate": metrics['success_rate'],
        "run_dir": run_dir,
        "training_time": total_time_str
    }

def set_seed(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main() -> None:
    """Main function to set up and run DQN training."""

    """
    Usage:
    
    Train vanilla DQN: python main.py --agent_type dqn
    Train Double DQN: python main.py --agent_type double_dqn
    Run both and compare: python main.py --run_both
    Compare existing results: python main.py --compare
    Compare specific results: python main.py --compare --dqn_path results/DQN_xxx/training_log.csv --double_dqn_path results/DOUBLE_DQN_yyy/training_log.csv
    """

    parser = argparse.ArgumentParser(description="Train a DQN agent on LunarLander-v3")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to train")
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
    parser.add_argument("--agent_type", type=str, default="dqn", choices=["dqn", "double_dqn"], 
                        help="Type of agent to use (dqn or double_dqn)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")

    # Comparison only arguments
    parser.add_argument("--compare", action="store_true", help="Compare DQN and Double DQN performance")
    parser.add_argument("--dqn_path", type=str, help="Path to DQN CSV log")
    parser.add_argument("--double_dqn_path", type=str, help="Path to Double DQN CSV log")
    parser.add_argument("--compare_output", type=str, default="results/comparison", help="Path to save comparison plot")

    # Run both agents for comparison
    parser.add_argument("--run_both", action="store_true",
                        help="Run both DQN and Double DQN and create comparison")

    args = parser.parse_args()

    # Set the seed if provided
    if args.seed is not None:
        print(f"Setting random seed to {args.seed}")
        set_seed(args.seed)

    # Set up device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set up environment parameters
    env = gym.make("LunarLander-v3", render_mode=None)  # No rendering during training
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    env.close()  # Close initial environment

    # Helper function to create and train an agent
    def create_and_train_agent(agent_type):
        """Create and train an agent of the specified type."""
        if agent_type == "dqn":
            print("\n=== Creating DQN Agent ===")
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
        else:  # double_dqn
            print("\n=== Creating Double DQN Agent ===")
            agent = DoubleDQNAgent(
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

        # Create fresh environment for training
        if args.seed is not None:
            training_env = gym.make("LunarLander-v3", render_mode=None)
            training_env.reset(seed=args.seed)
            training_env.action_space.seed(args.seed)
            training_env.observation_space.seed(args.seed)
        else:
            training_env = gym.make("LunarLander-v3", render_mode=None)

        # Train agent
        results = train(
            env=training_env,
            agent=agent,
            n_episodes=args.episodes,
            output_dir=args.output_dir,
            agent_type=agent_type.upper()
        )

        # Close environment after training
        training_env.close()

        print(f"\nTraining complete for {agent_type.upper()}!")
        print(f"Results saved to: {results['run_dir']}")
        print(f"Final average score: {results['final_avg_score']:.2f}")
        print(f"Final success rate: {results['final_success_rate']:.2f}%")
        print(f"Total training time: {results['training_time']}")

        return results

    # Handle all the different modes
    if args.run_both:

        if args.seed is not None:
            # Save states
            torch_state = torch.get_rng_state()
            torch_cuda_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
            np_state = np.random.get_state()
            random_state = random.getstate()
        
        # Run both agents and compare
        dqn_results = create_and_train_agent("dqn")

        if args.seed is not None:
            print(f"\nResetting random seed to {args.seed} for Double DQN training")
            # Restore states
            torch.set_rng_state(torch_state)
            if torch_cuda_state is not None:
                torch.cuda.set_rng_state(torch_cuda_state)
            np.random.set_state(np_state)
            random.setstate(random_state)
        
        double_dqn_results = create_and_train_agent("double_dqn")

        # Create comparison plots
        os.makedirs(args.compare_output, exist_ok=True)
        comparison_path = os.path.join(args.compare_output,
                                       f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

        from utils.train_logger import TrainLogger
        logger = TrainLogger()
        logger.plot_comparison(
            os.path.join(dqn_results["run_dir"], "training_log.csv"),
            os.path.join(double_dqn_results["run_dir"], "training_log.csv"),
            comparison_path,
            show_plot = False
        )

        print(f"\nComparison plot saved to: {comparison_path}")

        # Create a summary table
        # Create a summary table with better formatting
        print("\nSummary of performance over last 100 episodes:")
        print("+--------------------------+-------------------+-------------------+")
        print("| Metric                   | DQN (Vanilla)     | DQN + Double      |")
        print("+--------------------------+-------------------+-------------------+")
        print(
            f"| Average Episodic Reward | {dqn_results['final_avg_score']:<17.2f} | {double_dqn_results['final_avg_score']:<17.2f} |")
        print(
            f"| Average Return          | {dqn_results['final_avg_return']:<17.2f} | {double_dqn_results['final_avg_return']:<17.2f} |")
        print(
            f"| Success Rate (%)        | {dqn_results['final_success_rate']:<16.2f}% | {double_dqn_results['final_success_rate']:<16.2f}% |")
        print("+--------------------------+-------------------+-------------------+")
        if args.seed is not None:
            print(f"Runs both used seed: ({args.seed})")

        # Try to display the comparison plot
        try:
            if os.path.exists(comparison_path):
                img = plt.imread(comparison_path)
                plt.figure(figsize=(12, 18))
                plt.imshow(img)
                plt.axis('off')
                plt.show()
        except Exception as e:
            print(f"Could not display comparison plot: {e}")

    elif args.compare:
        # Compare existing results
        from utils.train_logger import TrainLogger

        if args.dqn_path and args.double_dqn_path:
            # Use provided paths
            dqn_csv = args.dqn_path
            double_dqn_csv = args.double_dqn_path
        else:
            # Find the most recent DQN and Double DQN results
            result_dirs = [d for d in os.listdir("results") if os.path.isdir(os.path.join("results", d))]

            dqn_dirs = sorted([d for d in result_dirs if d.startswith("DQN_")], reverse=True)
            double_dqn_dirs = sorted([d for d in result_dirs if d.startswith("DOUBLE_DQN_")], reverse=True)

            if not dqn_dirs or not double_dqn_dirs:
                print("Error: Could not find both DQN and Double DQN results. Run both types first.")
                return

            dqn_csv = os.path.join("results", dqn_dirs[0], "training_log.csv")
            double_dqn_csv = os.path.join("results", double_dqn_dirs[0], "training_log.csv")

        # Create comparison plots
        os.makedirs(args.compare_output, exist_ok=True)
        comparison_path = os.path.join(args.compare_output,
                                       f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

        logger = TrainLogger()
        logger.plot_comparison(dqn_csv, double_dqn_csv, comparison_path, show_plot=False)

        print(f"Comparison plot saved to: {comparison_path}")

        # Try to display the comparison plot
        try:
            if os.path.exists(comparison_path):
                img = plt.imread(comparison_path)
                plt.figure(figsize=(12, 18))
                plt.imshow(img)
                plt.axis('off')
                plt.show()
        except Exception as e:
            print(f"Could not display comparison plot: {e}")
    else:
        # Run a single agent (DQN or Double DQN)
        create_and_train_agent(args.agent_type)


if __name__ == "__main__":
    main()