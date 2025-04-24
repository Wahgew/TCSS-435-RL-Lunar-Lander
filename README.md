# Deep Q-Network for Lunar Lander

## Project Overview
This repository contains our team's implementation of Deep Q-Networks (DQN) for solving the LunarLander-v2 environment from Gymnasium. The project was developed as part of the TCSS 435 Reinforcement Learning course.

## Team Members
- Peter W Madin
- Andrew Hwang
- Sopheanith Ny
- Ken Egawa
- Mohammed Al-hamed

## Project Description
We implement a Deep Q-Network (DQN) to train an agent to successfully land a lunar module in the LunarLander-v2 environment. The project also includes an implementation of [Extension Name], which aims to improve upon the vanilla DQN algorithm.

### Environment
- LunarLander-v2 from Gymnasium
  - Continuous state space
  - Discrete action space (left, right, thruster)

### Requirements
- Python 3.10+
- PyTorch (or TensorFlow)
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

## Extension: [Extension Name]
For our extension, we implemented [brief description of your chosen extension]. This extension aims to [explain how it improves the vanilla DQN].

## Setup and Installation
```bash
# Clone the repository
git clone https://github.com/Wahgew/TCSS-435-RL-Lunar-Lander.git
cd dqn-lunar-lander

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
To train the agent:
```bash
python main.py --mode train --agent [vanilla_dqn/extension_dqn] --episodes 1000
```

To evaluate a trained agent:
```bash
python main.py --mode evaluate --agent [vanilla_dqn/extension_dqn] --model path/to/model
```

## Results
[This section will be updated with training results, performance metrics, and comparative analysis between vanilla DQN and our extension]

## References
- [Original DQN Paper](https://www.nature.com/articles/nature14236)
- [Extension Paper]
- [Any other relevant references]

## License
[Add your license information here]
