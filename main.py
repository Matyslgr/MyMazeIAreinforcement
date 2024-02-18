from envs.maze_env import MazeEnv
from envs.view import View
from AI import train, simulate
from matplotlib import pyplot as plt
import sys
import pygame
import time as tm

def print_usage():
    print("Usage: python main.py enable_render maze_size ([mode] or [enable_shortest_path])")
    print("enable_render: 0 or 1")
    print("maze_size: int")
    print("mode: normal or plus")
    print("enable_shortest_path: 0 or 1")

def show_reward(list_reward):
    plt.figure()
    plt.plot(list_reward)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Evolution of Total Reward during Training')
    plt.show()

def render_training(sleep_time):
    maze=env.maze
    view = View(maze_size=maze_size, screen_size=screen_size, maze_screen=maze_screen, maze=maze)
    list_reward = None
    Q = None
    while True:
        res = view.render()
        if res == "exit":
            break
        if res == "train":
            print("Training...")
            list_reward, Q = train(env, view, enable_render=enable_render)
        if res == "simulate":
            print("Simulating...")
            simulate(env, view, Q, sleep_time)
    view.quit_game()
    env.close()
    return list_reward, Q

def optimize_ai(sleep_time):
    maze=env.maze
    debut = tm.time()
    list_reward, Q = train(env, enable_render=enable_render)
    fin = tm.time()

    temps_execution = fin - debut
    view = View(maze_size=maze_size, screen_size=screen_size, maze_screen=maze_screen, maze=maze, optimise=True, time=temps_execution)
    while True:
        res = view.render()
        if res == "exit":
            break
        if res == "simulate":
            simulate(env, view, Q, sleep_time)
    view.quit_game()
    env.close()
    return list_reward, Q

def get_params(argc, argv):
    try:
        if argc == 3:
            enable_render = int(argv[1])
            maze_size = (int(argv[2]), int(argv[2]))
            mode = "normal"
            enable_shortest_path = False
        elif argc == 4:
            enable_render = int(argv[1])
            maze_size = (int(argv[2]), int(argv[2]))
            if argv[3] == "normal" or argv[3] == "plus":
                mode = argv[3]
                enable_shortest_path = None
            else:
                mode = "normal"
                enable_shortest_path = bool(int(argv[3]))
    except:
        print_usage()
        sys.exit(1)
    return enable_render, maze_size, mode, enable_shortest_path

# argv[1] is enable_render
# argv[2] is maze_size
# argv[3] is mode
# argv[4] is enable_shortest_path 

# or if argc == 2 and argv[1] is -h

if __name__ == "__main__":
    argv = sys.argv
    argc = len(argv)

    if argc == 2 and argv[1] == "-h":
        print_usage()
        sys.exit(0)
    enable_render, maze_size, mode, enable_shortest_path = get_params(argc, argv)
    pygame.init()
    maze_screen = (min(pygame.display.list_modes()[0]) * 0.8, min(pygame.display.list_modes()[0]) * 0.8)
    screen_size = (maze_screen[0] * 1.5, maze_screen[1])
    env = MazeEnv(maze_size=maze_size, mode=mode, enable_shortest_path=enable_shortest_path)

    if (min(maze_size) > 10):
        sleep_time = 0.05
    else:
        sleep_time = 0.2
    list_reward = None
    if not enable_render:
        list_reward, Q = optimize_ai(sleep_time)
    else:
        list_reward, Q = render_training(sleep_time)
    if list_reward is not None and maze_size[0] < 15 and maze_size[1] < 15:
        show_reward(list_reward)
    pygame.quit()