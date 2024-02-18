import sys
import numpy as np
import math
import random
import time as tm

# Listes Environments
# 'maze-v0'
# 'maze-sample-5x5-v0'
# 'maze-random-5x5-v0'
# 'maze-sample-10x10-v0'
# 'maze-random-10x10-v0'
# 'maze-sample-3x3-v0'
# 'maze-random-3x3-v0'
# 'maze-sample-100x100-v0'
# 'maze-random-100x100-v0'
# 'maze-random-10x10-plus-v0'
# 'maze-random-20x20-plus-v0'
# 'maze-random-30x30-plus-v0'

def global_init(env):
    global MAZE_SIZE
    global NUM_BUCKETS
    global MIN_EXPLORE_RATE
    global MIN_LEARNING_RATE
    global DECAY_FACTOR
    global NUM_ACTIONS
    global STATE_BOUNDS

    MAZE_SIZE = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    NUM_BUCKETS = MAZE_SIZE  # one bucket per grid
    MIN_EXPLORE_RATE = 0.001 # def 0.001
    MIN_LEARNING_RATE = 0.2
    DECAY_FACTOR = np.prod(MAZE_SIZE, dtype=float) / 10.0
    NUM_ACTIONS = env.action_space.n
    STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))

def select_action(env, state, explore_rate, Q):
    # Select a random action
    if random.random() < explore_rate:
        action = env.action_space.sample()
    # Select the action with the highest q
    else:
        action = int(np.argmax(Q[state]))
    return action


def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))


def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))


def state_to_bucket(state):
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i]-1)*STATE_BOUNDS[i][0]/bound_width
            scaling = (NUM_BUCKETS[i]-1)/bound_width
            bucket_index = int(round(scaling*state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)

def display_train(view, episode, position=None, total_reward=0):
    if view.maze_size[0] >= 15 and view.maze_size[1] >= 15 and episode % 100 == 0:
            view.render(position=position, total_reward=total_reward)
    else:
        view.render(position=position, total_reward=total_reward)
            
def train(env, view=None, enable_render=True):
    global_init(env)
    #Learning parameters
    NUM_EPISODES=50000
    MAX_T=1000
    STREAK_TO_END=100
    Q=np.zeros(NUM_BUCKETS + (NUM_ACTIONS,), dtype=float)
    learning_rate = get_learning_rate(0)
    explore_rate = get_explore_rate(0)
    discount_factor = 0.99
    num_streaks = 0
    list_reward = []
    find_solution = 0

    for episode in range(NUM_EPISODES):
        obv = env.reset()
        s = state_to_bucket(obv)
        total_reward = 0
        for t in range(MAX_T):
            a = select_action(env, s, explore_rate, Q)
            obv, r1, d, _ = env.step(a)
            if enable_render:
                position = (env.robot[0], env.robot[1])
                display_train(view, episode, position, total_reward=total_reward)
            
            s1 = state_to_bucket(obv)
            total_reward += r1

            best_q = np.amax(Q[s1])
            Q[s + (a,)] += learning_rate * (r1 + discount_factor * (best_q) - Q[s + (a,)])
            s = s1
            if (enable_render and view.game_over):
                sys.exit(0)
            if d:
                print(f"Episode {episode} finished after {t} time steps with total reward = {total_reward} (streak {num_streaks}).")
                num_streaks += 1
                find_solution += 1
                break
            elif t >= MAX_T - 1:
                num_streaks = 0
                print(f"Episode {episode} timed out at {t} with total reward = {total_reward}. (find exit {find_solution})")
            explore_rate = get_explore_rate(episode)
            learning_rate = get_learning_rate(episode)
        list_reward.append(total_reward)
        if enable_render:
            view.reset_path_matrix()
        if num_streaks > STREAK_TO_END:
            break
    return list_reward, Q

def simulate(env, view, Q, sleep_time=0.1):
    obv = env.reset()
    s = state_to_bucket(obv)
    d = False
    reward = 0
    time = 0

    view.path_matrix = np.zeros((env.maze_size[0], env.maze_size[1]), dtype=int)
    position = (env.robot[0], env.robot[1])
    view.render(position=position)
    tm.sleep(sleep_time)

    while not d:
        action = int(np.argmax(Q[s]))
        obv, r1, d, _ = env.step(action)
        position = (env.robot[0], env.robot[1])
        view.render(position=position)
        tm.sleep(sleep_time)
        if view.game_over:
            sys.exit(0)
        s1 = state_to_bucket(obv)
        s = s1
        reward += r1
        time += 1
    print("Simulation ended at time %d with total reward = %f." % (time, reward))
