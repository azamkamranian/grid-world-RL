import numpy as np
import gym


# Define useful functions

def to_bin(value, bins):
    return np.digitize(x=[value], bins=bins)[0]


def build_state(features):
    return int("".join(map(lambda feature: str(int(feature)), features)))


def transform(new_state):
    cart_pos, cart_vel, pole_angle, pole_vel = new_state
    return build_state([to_bin(cart_pos, cart_position_bins),
                        to_bin(cart_vel, cart_velocity_bins),
                        to_bin(pole_angle, pole_angle_bins),
                        to_bin(pole_vel, pole_velocity_bins)])


def random_action(a, eps=0.1):
    # we'll use epsilon-soft to ensure all states are visited
    # what happens if you don't do this? i.e. eps=0
    p = np.random.random()
    if p < (1 - eps):
        return a
    else:
        return np.random.choice([0, 1])


########################################################################
####  Q Learning with Q table and use Bins ############
####  optimal Q-table values for the problem ############
########################################################################

env = gym.make('CartPole-v0').env  # Create environment

## Number of states we build
bins = 9
cart_position_bins = np.linspace(-2.4, 2.4, bins)
cart_velocity_bins = np.linspace(-2, 2, bins)  # (-inf, inf) (I did not check that these were good values)
pole_angle_bins = np.linspace(-0.4, 0.4, bins)
pole_velocity_bins = np.linspace(-3, 3, bins)

# Number of possible actions
action_size = env.action_space.n

# Number of possible states
state_size = (bins + 1) ** env.observation_space.shape[0]

# Q-table
qtable = np.zeros((state_size, action_size))

episodes = 10000  # Total episodes
lr = 0.3  # Learning rate
decay_fac = 0.00001  # Decay learning rate each iteration
gamma = 0.90  # Discounting rate - later rewards impact less
number_of_test = 10

t = 1.0
t2 = 1.0
ALPHA = 0.1

# Exploration (Q-learning)
for episode in range(episodes):

    if episode % 100 == 0:
        t += 0.01
        t2 += 0.01
    lr = ALPHA / t2

    state = env.reset()  # Reset the environment
    done = False  # Are we done with the environment
    step = 0

    # lr -= decay_fac  # Decaying learning rate
    # if lr <= 0:  # Nothing more to learn?
    #     break

    action = np.argmax(qtable[transform(state), :])
    action = random_action(action, eps=0.5 / t)

    while not done:
        new_state, reward, done, info = env.step(action)

        new_action = np.argmax(qtable[transform(state), :])
        new_action = random_action(new_action, eps=0.5 / t)

        if done and step < 199:
            reward = -300

        qtable[transform(state), action] = qtable[transform(state), action] + lr * (
                reward + gamma * qtable[transform(new_state), new_action] - qtable[transform(state), action])

        # if done.. jump to next episode
        if done:
            break

        # moving states
        state = new_state
        action = new_action
        step += 1

    if (episode % 1000 == 0):
        print('episode = ', episode)
        print('learning rate = ', lr)
        print('-----------')

print("END TRAINING")

# Using Q-table for test
for _ in range(number_of_test):
    ## New environment
    state = env.reset()
    env.render()
    done = False
    total_reward = 0

    while (done == False):
        env.render()
        action = np.argmax(qtable[transform(state), :])  # Choose best action (Q-table)
        state, reward, done, info = env.step(action)  # Take action
        total_reward += reward  # Summing rewards

    print('Episode Reward = ', total_reward)
