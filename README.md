# Deep Q-Network for Lunar Lander

## Project Overview
This repository contains our team's implementation of Deep Q-Networks (DQN) for solving the LunarLander-v3 environment from Gymnasium. The project was developed as a project from TCSS 435: Artificial Intelligence And Knowledge Acquisition.

## Team Members
- Peter W Madin
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
pip install gymnasium
pip install gymnasium[box2d]
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
python main.py --agent_type dqn
python main.py --agent_type double_dqn
python main.py --run_both
```


## Results
[This section will be updated with training results, performance metrics, and comparative analysis between vanilla DQN and our extension]

## References
- [Original DQN Paper](https://www.nature.com/articles/nature14236)
- [Pytorch Reinforcement Learning (DQN) Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [Extension Paper](https://arxiv.org/abs/1509.06461)

## License
[Add your license information here]
