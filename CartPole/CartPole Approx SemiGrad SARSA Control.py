import numpy as np
import gym


# Define useful functions
class Model:
    def __init__(self):
        self.theta = np.random.randn(IDX) / np.sqrt(IDX)

    def sa2x(self, s, a):
        x = np.zeros(len(self.theta))
        idx = SA2IDX[s][a]
        x[idx] = 1
        return x

    def predict(self, s, a):
        x = self.sa2x(s, a)
        return self.theta.dot(x)

    def grad(self, s, a):
        return self.sa2x(s, a)


def getQs(model, s):
    # we need Q(s,a) to choose an action
    # i.e. a = argmax[a]{ Q(s,a) }
    Qs = [0, 0]
    for a in range(2):
        q_sa = model.predict(s, a)
        Qs[a] = q_sa
    return Qs


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
        return np.random.choice((0, 1))


########################################################################
####  Approximate SemiGradient SARSA Control #############
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

# Model
SA2IDX = np.arange(state_size * action_size).reshape(state_size, action_size)
IDX = state_size * action_size - 1

episodes = 10000  # Total episodes
lr = 0.3  # Learning rate
decay_fac = 0.00001  # Decay learning rate each iteration
gamma = 0.90  # Discounting rate - later rewards impact less
number_of_test = 10

t = 1.0
t2 = 1.0
model = Model()
ALPHA = 0.1

# Exploration (Q-learning)
for episode in range(episodes):
    step = 0

    if episode % 100 == 0:
        t += 0.01
        t2 += 0.01
    lr = ALPHA / t2

    state = env.reset()  # Reset the environment
    done = False  # Are we done with the environment

    lr -= decay_fac  # Decaying learning rate
    if lr <= 0:  # Nothing more to learn?
        break

    Qs = getQs(model, transform(state))
    action = np.argmax(Qs)
    # use epsilon greedy
    action = random_action(action, eps=0.5 / t)

    while not done:
        new_state, reward, done, info = env.step(action)

        old_theta = model.theta.copy()
        if done and step < 199:
            reward = -300
            model.theta += lr * (reward - model.predict(transform(state), action)) * model.grad(transform(state),
                                                                                                action)
        elif done and step >= 199:
            model.theta += lr * (reward - model.predict(transform(state), action)) * model.grad(transform(state),
                                                                                                action)
        else:
            # not terminal
            Qs2 = getQs(model, transform(new_state))
            new_action = np.argmax(Qs2)
            new_action = random_action(new_action, eps=0.5 / t)

            # we will update model theta as we exprience
            model.theta += lr * (reward + gamma * model.predict(transform(new_state), new_action) - model.predict(
                transform(state), action)) * model.grad(transform(state), action)

            # next state becomes current state
            state = new_state
            action = new_action
            step += 1

        # if done.. jump to next episode
        if done:
            break

    if (episode % 1000 == 0):
        print('episode = ', episode)
        print('learning rate = ', lr)
        print('-----------')

print("END TRAINING")

# Using Trained MOdel for test
for _ in range(number_of_test):
    ## New environment
    state = env.reset()
    env.render()
    done = False
    total_reward = 0

    while (done == False):
        env.render()
        Qs = getQs(model, transform(state))
        action = np.argmax(Qs)
        state, reward, done, info = env.step(action)  # Take action
        total_reward += reward  # Summing rewards

    print('Episode Reward = ', total_reward)
