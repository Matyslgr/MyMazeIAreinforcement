# MyMazeReinforcement

## Overview

This project involves simulating maze-solving using reinforcement learning. The maze environment is represented as a grid, and an artificial intelligence agent learns to navigate through the maze to reach the goal. The project includes visualization tools to observe the training process and simulate the agent's behavior.

## Prerequisites

Before running the project, make sure you have the following dependencies installed:

- Python (>=3.6)
- Pygame library
- Matplotlib library
- Gym

You can install the required libraries using the following command:

```bash
pip install pygame matplotlib gym
```

## Usage

### Command Line Interface

To run the project, use the main.py script. The basic usage is as follows:
- enable_render: 0 or 1, indicating whether to enable rendering during training.
- maze_size: an integer specifying the size of the maze (width and height).
- mode: optional parameter, either "normal" or "plus" (default is "normal").
- enable_shortest_path: optional parameter, 0 or 1, indicating whether to enable visualization of the shortest path (default is False).

### Exemples
```bash
# Basic usage
python3 main.py 1 10

# Enable rendering and set the maze size to 15
python3 main.py 1 15

# Enable rendering, set the maze size to 12, and use the "plus" mode
python3 main.py 1 12 plus

# Disable rendering, set the maze size to 20, and enable visualization of the shortest path
python3 main.py 0 20 1
```

### Visualization
During training, the script provides an interactive interface to render the maze and visualize the agent's progress.
Press 'train' to start the training process and 'simulate' to observe the agent navigating the maze. Use the 'exit' option to close the visualization.

## Project Structure

- envs: Contains the maze environment and visualization components.
- AI: Implements reinforcement learning algorithms for agent training.
- main.py: The main script to run the project and configure parameters.
- README.md: This readme file providing information about the project.
