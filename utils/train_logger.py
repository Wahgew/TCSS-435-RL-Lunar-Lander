# utils/train_logger.py
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from datetime import datetime


class TrainLogger:
    """
    Logger for tracking, recording, and visualizing training metrics.

    This class is specifically designed to fulfill the requirements of Part 3
    of the assignment, which requires tracking and comparing:
    - Total episodic reward
    - Return G (discounted rewards)
    - Landing success rates
    """

    def __init__(self) -> None:
        """Initialize a TrainLogger to track metrics across episodes."""
        # Raw metrics for each episode
        self.scores: List[float] = []  # Total undiscounted reward per episode
        self.returns: List[float] = []  # Discounted return (G) per episode
        self.successes: List[bool] = []  # Binary landing success per episode

        # Moving averages (for smoother visualization)
        self.avg_scores: List[float] = []  # Moving average of scores
        self.avg_returns: List[float] = []  # Moving average of returns
        self.success_rates: List[float] = []  # Success rate as percentage

        # Episode numbers for plotting
        self.episodes: List[int] = []

    def log_episode(self, episode: int, score: float, episode_return: float, success: bool,
                    window_size: int = 100) -> Dict[str, float]:
        """
        Record metrics for a completed episode and calculate moving averages.

        Args:
            episode: Current episode number
            score: Total reward received in the episode
            episode_return: Discounted return (G) for the episode
            success: Whether the landing was successful (binary)
            window_size: Window size for calculating moving averages (100 as per assignment)

        Returns:
            Dictionary with current metrics including moving averages
        """
        # Store raw metrics
        self.scores.append(score)
        self.returns.append(episode_return)
        self.successes.append(success)
        self.episodes.append(episode)

        # Calculate moving averages over last window_size episodes
        # This will be used for the "over the last 100 episodes" metrics in Part 3
        window_start = max(0, len(self.scores) - window_size)
        avg_score = np.mean(self.scores[window_start:])
        avg_return = np.mean(self.returns[window_start:])
        success_rate = np.mean(self.successes[window_start:]) * 100  # Convert to percentage

        # Store the moving averages
        self.avg_scores.append(avg_score)
        self.avg_returns.append(avg_return)
        self.success_rates.append(success_rate)

        # Return current metrics (useful for progress tracking during training)
        return {
            'episode': episode,
            'score': score,
            'return': episode_return,
            'success': success,
            'avg_score': avg_score,
            'avg_return': avg_return,
            'success_rate': success_rate
        }

    def plot_training(self, save_path: Optional[str] = None) -> None:
        """
        Generate the required plots for Part 3 of the assignment.

        Creates three plots as specified in the assignment:
        1. Episodic Reward vs. Episode Number
        2. Episodic Return vs. Episode Number
        3. Success Rate over Time

        Args:
            save_path: Path to save the plot; if None, the plot is displayed
        """
        # Create a figure with three subplots (one for each required metric)
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))

        # Plot 1: Episodic Reward vs. Episode Number (required in assignment)
        axes[0].plot(self.episodes, self.scores, alpha=0.3, color='blue', label='Per Episode')
        axes[0].plot(self.episodes, self.avg_scores, color='blue', label='Moving Avg')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Score')
        axes[0].set_title('Episodic Reward vs. Episode Number')
        axes[0].grid(True)
        axes[0].legend()

        # Plot 2: Episodic Return vs. Episode Number (required in assignment)
        axes[1].plot(self.episodes, self.returns, alpha=0.3, color='green', label='Per Episode')
        axes[1].plot(self.episodes, self.avg_returns, color='green', label='Moving Avg')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Return')
        axes[1].set_title('Episodic Return vs. Episode Number')
        axes[1].grid(True)
        axes[1].legend()

        # Plot 3: Success Rate over Time (required in assignment)
        axes[2].plot(self.episodes, [int(s) for s in self.successes], 'o', alpha=0.3, color='red',
                     label='Success (0/1)')
        axes[2].plot(self.episodes, [s / 100 for s in self.success_rates], color='red', label='Success Rate')
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('Success')
        axes[2].set_title('Success Rate over Time')
        axes[2].set_ylim([-0.1, 1.1])  # Binary plot range with small margin
        axes[2].grid(True)
        axes[2].legend()

        # Adjust spacing between subplots
        plt.tight_layout()

        # Save or display the figure
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Training plots saved to {save_path}")
        else:
            plt.show()

    def save_to_csv(self, filepath: str) -> None:
        """
        Save all training metrics to a CSV file for further analysis.

        This data can be used to create the summary table required in Part 3.

        Args:
            filepath: Path where the CSV file will be saved
        """
        # Create a DataFrame with all metrics
        df = pd.DataFrame({
            'episode': self.episodes,
            'score': self.scores,
            'return': self.returns,
            'success': self.successes,
            'avg_score': self.avg_scores,
            'avg_return': self.avg_returns,
            'success_rate': self.success_rates
        })

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save to CSV
        df.to_csv(filepath, index=False)
        print(f"Training data saved to {filepath}")

    def get_final_metrics(self, num_episodes: int = 100) -> Dict[str, float]:
        """
        Get average metrics over the last N episodes.

        This is useful for generating the summary table required in Part 3,
        which asks for metrics averaged over the last 100 episodes.

        Args:
            num_episodes: Number of recent episodes to average over (default: 100)

        Returns:
            Dictionary with average metrics over the specified number of episodes
        """
        if len(self.scores) < num_episodes:
            return {
                'avg_score': np.mean(self.scores),
                'avg_return': np.mean(self.returns),
                'success_rate': np.mean(self.successes) * 100
            }

        # Get metrics for the last num_episodes
        recent_scores = self.scores[-num_episodes:]
        recent_returns = self.returns[-num_episodes:]
        recent_successes = self.successes[-num_episodes:]

        return {
            'avg_score': np.mean(recent_scores),
            'avg_return': np.mean(recent_returns),
            'success_rate': np.mean(recent_successes) * 100
        }