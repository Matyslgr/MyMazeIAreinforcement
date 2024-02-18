import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from envs.maze import Maze
import os

class MazeEnv(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    ACTION = ["N", "S", "E", "W"]

    def __init__(self, maze_file=None, maze_size=None, mode=None, enable_shortest_path=False):

        self.viewer = None

        if mode == "plus":
            has_loops = True
            num_portals = int(round(min(maze_size)/4))
        else:
            has_loops = False
            num_portals = 0

        self.maze_size = maze_size

        # forward or backward in each dimension
        self.action_space = spaces.Discrete(2*len(self.maze_size))

        # observation is the x, y coordinate of the grid
        low = np.zeros(len(self.maze_size), dtype=int)
        high =  np.array(self.maze_size, dtype=int) - np.ones(len(self.maze_size), dtype=int)
        self.observation_space = spaces.Box(low, high, dtype=np.int64)

        # initial condition
        self.state = None
        self.steps_beyond_done = None

        # Simulation related variables.
        self.seed()
        self.reset()

        # Load a maze
        if maze_file is None:
            self.__maze = Maze(maze_size=maze_size, has_loops=has_loops, num_portals=num_portals, enable_shortest_path=enable_shortest_path)
        else:
            if not os.path.exists(maze_file):
                dir_path = os.path.dirname(os.path.abspath(__file__))
                rel_path = os.path.join(dir_path, "maze_samples", maze_file)
                if os.path.exists(rel_path):
                    maze_file = rel_path
                else:
                    raise FileExistsError("Cannot find %s." % maze_file)
            self.__maze = Maze(maze_cells=Maze.load_maze(maze_file), has_loops=has_loops, num_portals=num_portals, enable_shortest_path=enable_shortest_path)

        self.maze_size = self.__maze.maze_size
        
        # Set the starting point
        self.__entrance = np.zeros(2, dtype=int)

        # Set the Goal
        self.__goal = np.array(self.maze_size) - np.array((1, 1))

        # Create the Robot
        self.__robot = self.__entrance

    def move_robot(self, dir):
        if dir not in self.__maze.COMPASS.keys():
            raise ValueError("dir cannot be %s. The only valid dirs are %s."
                             % (str(dir), str(self.__maze.COMPASS.keys())))

        if self.__maze.is_open(self.__robot, dir):

            # move the robot
            self.__robot += np.array(self.__maze.COMPASS[dir])

            # if it's in a portal afterward
            if self.maze.is_portal(self.robot):
                self.__robot = np.array(self.maze.get_portal(tuple(self.robot)).teleport(tuple(self.robot)))
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        if isinstance(action, int):
            self.move_robot(self.ACTION[action])
        else:
            self.move_robot(action)

        if np.array_equal(self.robot, self.goal):
            reward = 1
            done = True
        else:
            reward = -0.1/(self.maze_size[0]*self.maze_size[1])
            done = False

        self.state = self.robot

        info = {}

        return self.state, reward, done, info

    def reset_robot(self):
        self.__robot = np.zeros(2, dtype=int)

    def reset(self):
        self.reset_robot()
        self.state = np.zeros(2)
        self.steps_beyond_done = None
        self.done = False
        return self.state

    def is_game_over(self):
        return self.__game_over

    @property
    def maze(self):
        return self.__maze

    @property
    def robot(self):
        return self.__robot

    @property
    def entrance(self):
        return self.__entrance

    @property
    def goal(self):
        return self.__goal