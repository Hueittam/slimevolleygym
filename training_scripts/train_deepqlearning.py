import gym
import numpy as np
import slimevolleygym
from training_scripts.qlearning.qlearning import Agent
from training_scripts.qlearning.utils import plot_learning_curve

if __name__ == "__main__":
    env = gym.make("SlimeVolley-v0")
    action_space = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
    ]
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=6, eps_end=0.01, input_dims=[12],
                  learning_rate=0.03)
    scores, eps_history = [], []
    n_games = 100

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action_number = agent.choose_action(observation)
            action = action_space[action_number]
            print("action:", action_number)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action_number, reward, observation_, done)
            agent.learn()
            observation = observation_

        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        print(f"episode: {i}, score: {score}, avg_score: {avg_score}, epsilon: {agent.epsilon:.2f}")
        x = [i + 1 for i in range(n_games)]
        filename = "slimevolley.png"
        plot_learning_curve(x, scores, eps_history, filename)
