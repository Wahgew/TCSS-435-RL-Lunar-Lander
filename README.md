# Deep Q-Network for Lunar Lander
## Project Overview
This repository contains our team's implementation of Deep Q-Networks (DQN) for solving the [LunarLander-v3](https://gymnasium.farama.org/environments/box2d/lunar_lander/) environment from Gymnasium. The project was developed as a project from TCSS 435: Artificial Intelligence & Knowledge Acquisition.
## Team Members
- Peter Wahyudianto Madin
- Andrew Hwang
- Sopheanith Ny
- Ken Egawa
- Mohammed Al-hamed

## Project Description
We implement a Deep Q-Network (DQN) to train an agent to successfully land a lunar module in the LunarLander-v3 environment. The project also includes an implementation of Double DQN, which aims to improve upon the vanilla DQN algorithm.

### Environment
- LunarLander-v3 from Gymnasium
  - Continuous state space
  - Discrete action space (left, right, thruster)
- Python 3.10+
- PyTorch 
- Gymnasium
- Matplotlib
- NumPy

## Project Structure
```
.
├── agents/
│   └── dqn_agent.py         # DQN Agent implementation
├── networks/
│   └── q_network.py         # Neural network architecture
├── utils/
│   ├── replay_buffer.py     # Experience replay implementation
│   └── train_logger.py      # Training metrics logger
├── results/                 # Saved models and training plots
├── main.py                  # Entry point for training
└── README.md
```

## DQN Implementation
Our implementation includes:
- Q-network with fully connected layers
- Target network (updated periodically)
- Experience replay buffer
- Epsilon-greedy exploration
- Online training loop

## Extension: [Double DQN](https://arxiv.org/abs/1509.06461)
For our extension, we implemented Double DQN, comparing it to our standard DQN implementation. This extension aims to reduce the impact of overestimations from the standard DQN algorithm through using different networks, 
online to select the best action and target to evaluate the actions value, this is in comparison to standard DQN, which uses the same network to select and evaluate.

## Setup and Installation
```bash
# Clone the repository
git clone https://github.com/Wahgew/TCSS-435-RL-Lunar-Lander.git
# Install required packages
All Packages: pip install torch matplotlib numpy pandas "gymnasium[box2d]"
- pip install torch
- pip install matplotlib
- pip install numpy
- pip install pandas
- pip install gymnasium[box2d]
```

## Notes
- If installing `gymnasium[box2d]` fails with a wheel build error, you may need to install **SWIG**:
  ```bash
  choco install swig -y
- If you do not have Chocolatey installed (`choco` command not found), you will need to install it first.
You can find the installation guide here: https://chocolatey.org/install

## Usage
To train the agent (plots + statistics on completion):
```bash
# Train and run DQN agent.
python main.py --agent_type dqn
#Train and run double DQN agent.
python main.py --agent_type double_dqn
#Train and run both DQN and double DQN sequentially.
python main.py --run_both
# Use arg parameter below to specify a seed, default is None.
--seed
```

## Results 
Trained on python main.py --run_both --seed 123

![Training Performance Comparison](https://github.com/user-attachments/assets/a6ffaf96-c68e-4d65-bf68-168900efaaac)

The graph above shows the comparison of training performance between DQN and Double DQN algorithms. The results demonstrate that Double DQN reaches stable performance faster and achieves higher average rewards over time compared to standard DQN.

### DQN Agent Landing Performance
[View DQN Landing Video](https://github.com/user-attachments/assets/9d3f56df-fb28-406b-a6bc-61d315c9d93c)

The standard DQN agent demonstrates effective learning capabilities, eventually landing the module with some control, though with occasional instability in its approach.

### Double DQN Agent Landing Performance
[View Double DQN Landing Video](https://github.com/user-attachments/assets/4a2116c1-fcb4-4da1-9ff5-7b27548fbd15)

The Double DQN agent shows improved stability and efficiency in landing the lunar module, with more precise control of thrusters and smoother descent trajectories. This improvement aligns with the performance metrics shown in the graph, confirming Double DQN's advantage in reducing overestimation bias.

## References
- [Original DQN Paper](https://www.nature.com/articles/nature14236)
- [Pytorch Reinforcement Learning (DQN) Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [Extension Paper](https://arxiv.org/abs/1509.06461)
- [Reproducibility Pytorch](https://pytorch.org/docs/stable/notes/randomness.html)

## License
See the [LICENSE](LICENSE.md) file for license rights and limitations (MIT).
