import matplotlib.pyplot as plt

def plot_reward(reward_history,reward_history_averaged):
    plt.figure()
    plt.plot(reward_history, label="Reward")
    plt.plot(reward_history_averaged, label="Average reward over 10 episodes")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.show()
