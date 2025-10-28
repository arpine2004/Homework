"""
  Run this file at first, in order to see what is it printing. Instead of the print() use the respective log level
"""

############################### LOGGER
from abc import ABC, abstractmethod
from loguru import logger
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib
import random
matplotlib.use('TkAgg')


class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass


# --------------------------------------#


class Visualization():

    def plot1(self, rewards, num_trials=20000):
        # Visualize the performance of each bandit: linear and log
        """
        Visualizes the performance of the epsilon-greedy algorithm using both linear and logarithmic scales.

        :param rewards: Array of rewards from the experiment
        :param num_trials: Total number of trials (default is 20000)
        """

        cumulative_rewards = np.cumsum(rewards)

        # Linear plot for cumulative rewards
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(cumulative_rewards, label="Cumulative Rewards")
        plt.title("Cumulative Rewards (Linear Scale)")
        plt.xlabel("Number of Trials")
        plt.ylabel("Cumulative Reward")
        plt.grid(True)
        plt.legend()

        # Logarithmic plot for cumulative rewards
        plt.subplot(1, 2, 2)
        plt.plot(cumulative_rewards, label="Cumulative Rewards")
        plt.yscale('log')
        plt.title("Cumulative Rewards (Logarithmic Scale)")
        plt.xlabel("Number of Trials")
        plt.ylabel("Cumulative Reward (Log Scale)")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot2(self, eg_rewards, ts_rewards, num_trials=20000, Bandit_Reward=[1, 2, 3, 4]):
        # Compare E-greedy and thompson sampling cumulative rewards

        eg_cumulative_rewards = np.cumsum(eg_rewards)
        ts_cumulative_rewards = np.cumsum(ts_rewards)

        optimal_reward = np.max(Bandit_Reward)
        eg_cumulative_regret = optimal_reward * np.arange(1, num_trials + 1) - eg_cumulative_rewards
        ts_cumulative_regret = optimal_reward * np.arange(1, num_trials + 1) - ts_cumulative_rewards

        plt.figure(figsize=(12, 6))

        # Cumulative rewards plot
        plt.subplot(1, 2, 1)
        plt.plot(eg_cumulative_rewards, label="E-greedy Cumulative Rewards")
        plt.plot(ts_cumulative_rewards, label="Thompson Sampling Cumulative Rewards")
        plt.title("Cumulative Rewards Comparison")
        plt.xlabel("Number of Trials")
        plt.ylabel("Cumulative Reward")
        plt.grid(True)
        plt.legend()

        # Cumulative regrets plot
        plt.subplot(1, 2, 2)
        plt.plot(eg_cumulative_regret, label="E-greedy Cumulative Regret")
        plt.plot(ts_cumulative_regret, label="Thompson Sampling Cumulative Regret")
        plt.title("Cumulative Regret Comparison")
        plt.xlabel("Number of Trials")
        plt.ylabel("Cumulative Regret")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()


# --------------------------------------#

class EpsilonGreedy(Bandit):
    """
    Epsilon-Greedy algorithm implementation for multi-armed bandit problem.

    :param p: Array of true reward values for each arm
    """

    def __init__(self, p):
        """
        Initialize the EpsilonGreedy bandit.

        :param p: Array of true reward values for each arm
        """
        self.p = p
        self.p_estimates = np.zeros(len(p))
        self.N = np.zeros(len(p))

    def __repr__(self):
        """
        String representation of the EpsilonGreedy bandit.

        :return: String describing the bandit
        """
        return f"EpsilonGreedy Bandit"

    def pull(self, epsilon):
        """
        Pull the arm based on epsilon-greedy strategy.

        :param epsilon: Current epsilon value for exploration probability
        :return: The index of the selected arm
        """
        if np.random.random() < epsilon:  # Exploration
            return np.random.choice(len(self.p))  # Choose random arm
        else:  # Exploitation
            return np.argmax(self.p_estimates)  # Choose arm with the highest estimated reward

    def update(self, arm, reward):
        """
        Update the estimated reward for the selected arm.

        :param arm: The index of the arm that was pulled
        :param reward: The reward received from pulling the arm
        """
        self.N[arm] += 1
        self.p_estimates[arm] = ((self.N[arm] - 1) * self.p_estimates[arm] + reward) / self.N[arm]

    def experiment(self, num_trials):
        """
        Run the epsilon-greedy experiment for a specified number of trials.

        :param num_trials: Number of trials to run
        :return: Tuple containing (rewards, arms_selected, num_times_explored, num_times_exploited, num_optimal)
        """
        rewards = np.zeros(num_trials)
        arms_selected = np.zeros(num_trials, dtype=int)
        num_times_explored = 0
        num_times_exploited = 0
        num_optimal = 0
        optimal_arm = np.argmax(self.p)

        for t in range(1, num_trials + 1):
            epsilon = 1.0 / t

            arm = self.pull(epsilon)
            arms_selected[t - 1] = arm
            reward = self.p[arm]

            rewards[t - 1] = reward
            self.update(arm, reward)

            if arm == optimal_arm:
                num_optimal += 1

            if np.random.random() < epsilon:
                num_times_explored += 1
            else:
                num_times_exploited += 1

        return rewards, arms_selected, num_times_explored, num_times_exploited, num_optimal

    def report(self, rewards, arms_selected, num_times_explored, num_times_exploited, num_trials=20000,
               filename="bandit_results.csv"):
        """
        Report the results:
        - Store data in CSV with format {Bandit, Reward, Algorithm}
        - Print cumulative reward
        - Print cumulative regret

        :param rewards: Array of rewards from the experiment
        :param arms_selected: Array of arms selected in each trial
        :param num_times_explored: Number of times exploration was used
        :param num_times_exploited: Number of times exploitation was used
        :param num_trials: Total number of trials (default is 20000)
        :param filename: Name of the CSV file to save results
        """
        total_reward = np.sum(rewards)
        total_regret = (np.max(self.p) * num_trials) - total_reward

        logger.info(f"Cumulative Reward: {total_reward:.2f}")
        logger.info(f"Cumulative Regret: {total_regret:.2f}")
        logger.info(f"Total Reward Earned: {total_reward}")
        logger.info(f"Optimal arm selected {np.sum(rewards == np.max(self.p))} times")
        logger.debug(f"Number of times explored: {num_times_explored}")
        logger.debug(f"Number of times exploited: {num_times_exploited}")

        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Bandit", "Reward", "Algorithm"])

            for i in range(num_trials):
                writer.writerow([arms_selected[i], rewards[i], "EpsilonGreedy"])


# --------------------------------------#
class ThompsonSampling(Bandit):
    """
    Thompson Sampling algorithm implementation for multi-armed bandit problem.

    :param p: True rewards for each arm
    :param known_precision: Known precision (1 / variance) for the rewards
    :param prior_mean: The prior mean (initial belief) for each arm
    :param prior_variance: The prior variance (initial belief) for each arm
    """

    def __init__(self, p, known_precision=10.0, prior_mean=2.5, prior_variance=1):
        """
        Initialize the Thompson Sampling algorithm.

        :param p: True rewards for each arm ([1, 2, 3, 4])
        :param known_precision: Known precision (1 / variance) for the rewards
        :param prior_mean: The prior mean (initial belief) for each arm
        :param prior_variance: The prior variance (initial belief) for each arm
        """
        self.p = p
        self.known_precision = known_precision
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance

        self.arm_posterior_means = np.zeros(len(p))
        self.arm_posterior_variances = np.ones(len(p)) * prior_variance
        self.num_pulls = np.zeros(len(p))
        self.reward_sums = np.zeros(len(p))

    def __repr__(self):
        """
        String representation of the ThompsonSampling bandit.

        :return: String describing the bandit
        """
        return f"ThompsonSampling Bandit with known precision={self.known_precision}"

    def pull(self):
        """
        Pull an arm based on Thompson Sampling.
        Sample from the posterior distribution of each arm and select the one with the highest sample.

        :return: The index of the arm selected
        """
        # Sample from the posterior of each arm
        sampled_means = np.random.normal(self.arm_posterior_means, np.sqrt(self.arm_posterior_variances))

        # Select the arm with the highest sampled mean
        return np.argmax(sampled_means)

    def update(self, arm, reward):
        """
        Update the posterior mean and variance for the selected arm using the observed reward.

        :param arm: The arm selected in this trial
        :param reward: The reward observed from the selected arm
        """
        self.num_pulls[arm] += 1
        self.reward_sums[arm] += reward

        n = self.num_pulls[arm]

        tau_0 = 1 / self.prior_variance
        tau = self.known_precision
        tau_n = tau_0 + n * tau

        updated_variance = 1 / tau_n

        updated_mean = (tau_0 * self.prior_mean + tau * self.reward_sums[arm]) / tau_n

        self.arm_posterior_means[arm] = updated_mean
        self.arm_posterior_variances[arm] = updated_variance

    def experiment(self, num_trials=20000):
        """
        Run the Thompson Sampling experiment for a specified number of trials.

        :param num_trials: Number of trials to run (default is 20000)
        :return: Tuple containing (rewards, arms_selected, num_times_optimal)
        """
        rewards = np.zeros(num_trials)
        arms_selected = np.zeros(num_trials, dtype=int)
        num_times_optimal = 0
        optimal_arm = np.argmax(self.p)

        for t in range(1, num_trials + 1):
            arm = self.pull()
            arms_selected[t - 1] = arm
            reward = self.p[arm]

            rewards[t - 1] = reward
            self.update(arm, reward)

            if arm == optimal_arm:
                num_times_optimal += 1

        return rewards, arms_selected, num_times_optimal

    def report(self, rewards, arms_selected, num_times_optimal, num_trials=20000, filename="thompson_results.csv"):
        """
        Report the results:
        - Store data in CSV with format {Bandit, Reward, Algorithm}
        - Print cumulative reward
        - Print cumulative regret

        :param rewards: Array of rewards from the experiment
        :param arms_selected: Array of arms selected in each trial
        :param num_times_optimal: Number of times the optimal arm was selected
        :param num_trials: Total number of trials (default is 20000)
        :param filename: Name of the CSV file to save results
        """
        total_reward = np.sum(rewards)
        total_regret = (np.max(self.p) * num_trials) - total_reward

        logger.info(f"Cumulative Reward: {total_reward:.2f}")
        logger.info(f"Cumulative Regret: {total_regret:.2f}")
        logger.info(f"Total Reward Earned: {total_reward}")
        logger.info(f"Optimal arm selected {num_times_optimal} times")

        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Bandit", "Reward", "Algorithm"])

            for i in range(num_trials):
                writer.writerow([arms_selected[i], rewards[i], "ThompsonSampling"])


def comparison(eg_bandit, ts_bandit, eg_rewards, ts_rewards, eg_arms, ts_arms,
               num_trials=20000, Bandit_Reward=[1, 2, 3, 4]):
    """
    Compare the performances of Epsilon-Greedy and Thompson Sampling algorithms VISUALLY.

    :param eg_bandit: EpsilonGreedy object after experiment
    :param ts_bandit: ThompsonSampling object after experiment
    :param eg_rewards: Array of rewards from Epsilon-Greedy experiment
    :param ts_rewards: Array of rewards from Thompson Sampling experiment
    :param eg_arms: Array of arms selected in Epsilon-Greedy experiment
    :param ts_arms: Array of arms selected in Thompson Sampling experiment
    :param num_trials: Total number of trials
    :param Bandit_Reward: True reward values for each arm
    """

    fig = plt.figure(figsize=(16, 12))

    optimal_reward = np.max(Bandit_Reward)
    optimal_arm = np.argmax(Bandit_Reward)

    # 1. Cumulative Rewards Comparison
    ax1 = plt.subplot(3, 3, 1)
    eg_cumulative = np.cumsum(eg_rewards)
    ts_cumulative = np.cumsum(ts_rewards)
    ax1.plot(eg_cumulative, label='Epsilon-Greedy', alpha=0.8)
    ax1.plot(ts_cumulative, label='Thompson Sampling', alpha=0.8)
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('Cumulative Reward')
    ax1.set_title('Cumulative Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Cumulative Regret Comparison
    ax2 = plt.subplot(3, 3, 2)
    eg_regret = optimal_reward * np.arange(1, num_trials + 1) - eg_cumulative
    ts_regret = optimal_reward * np.arange(1, num_trials + 1) - ts_cumulative
    ax2.plot(eg_regret, label='Epsilon-Greedy', alpha=0.8)
    ax2.plot(ts_regret, label='Thompson Sampling', alpha=0.8)
    ax2.set_xlabel('Trial')
    ax2.set_ylabel('Cumulative Regret')
    ax2.set_title('Cumulative Regret')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Average Reward Over Time (Moving Average)
    ax3 = plt.subplot(3, 3, 3)
    window = 1000
    eg_moving_avg = np.convolve(eg_rewards, np.ones(window) / window, mode='valid')
    ts_moving_avg = np.convolve(ts_rewards, np.ones(window) / window, mode='valid')
    ax3.plot(range(window - 1, num_trials), eg_moving_avg, label='Epsilon-Greedy', alpha=0.8)
    ax3.plot(range(window - 1, num_trials), ts_moving_avg, label='Thompson Sampling', alpha=0.8)
    ax3.axhline(y=optimal_reward, color='r', linestyle='--', label='Optimal', alpha=0.5)
    ax3.set_xlabel('Trial')
    ax3.set_ylabel('Average Reward')
    ax3.set_title(f'Moving Average Reward (window={window})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Arm Selection Frequency Over Time (Epsilon-Greedy)
    ax4 = plt.subplot(3, 3, 4)
    window_size = 2000
    for arm in range(len(Bandit_Reward)):
        arm_selected = (eg_arms == arm).astype(int)
        arm_freq = np.convolve(arm_selected, np.ones(window_size) / window_size, mode='valid')
        label = f'Arm {arm} (r={Bandit_Reward[arm]})'
        if arm == optimal_arm:
            label += ' ★'
        ax4.plot(range(window_size - 1, num_trials), arm_freq, label=label, alpha=0.8)
    ax4.set_xlabel('Trial')
    ax4.set_ylabel('Selection Frequency')
    ax4.set_title('Epsilon-Greedy: Arm Selection Over Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Arm Selection Frequency Over Time (Thompson Sampling)
    ax5 = plt.subplot(3, 3, 5)
    for arm in range(len(Bandit_Reward)):
        arm_selected = (ts_arms == arm).astype(int)
        arm_freq = np.convolve(arm_selected, np.ones(window_size) / window_size, mode='valid')
        label = f'Arm {arm} (r={Bandit_Reward[arm]})'
        if arm == optimal_arm:
            label += ' ★'
        ax5.plot(range(window_size - 1, num_trials), arm_freq, label=label, alpha=0.8)
    ax5.set_xlabel('Trial')
    ax5.set_ylabel('Selection Frequency')
    ax5.set_title('Thompson Sampling: Arm Selection Over Time')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Total Arm Selection Distribution
    ax6 = plt.subplot(3, 3, 6)
    eg_counts = [np.sum(eg_arms == arm) for arm in range(len(Bandit_Reward))]
    ts_counts = [np.sum(ts_arms == arm) for arm in range(len(Bandit_Reward))]
    x = np.arange(len(Bandit_Reward))
    width = 0.35
    ax6.bar(x - width / 2, eg_counts, width, label='Epsilon-Greedy', alpha=0.8)
    ax6.bar(x + width / 2, ts_counts, width, label='Thompson Sampling', alpha=0.8)
    ax6.set_xlabel('Arm')
    ax6.set_ylabel('Number of Selections')
    ax6.set_title('Total Arm Selections')
    ax6.set_xticks(x)
    ax6.set_xticklabels([f'Arm {i}\n(r={Bandit_Reward[i]})' for i in range(len(Bandit_Reward))])
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')

    # 7. Regret Rate Over Time
    ax7 = plt.subplot(3, 3, 7)
    window = 1000
    eg_regret_rate = np.convolve(optimal_reward - eg_rewards, np.ones(window) / window, mode='valid')
    ts_regret_rate = np.convolve(optimal_reward - ts_rewards, np.ones(window) / window, mode='valid')
    ax7.plot(range(window - 1, num_trials), eg_regret_rate, label='Epsilon-Greedy', alpha=0.8)
    ax7.plot(range(window - 1, num_trials), ts_regret_rate, label='Thompson Sampling', alpha=0.8)
    ax7.set_xlabel('Trial')
    ax7.set_ylabel('Average Regret per Trial')
    ax7.set_title(f'Regret Rate (window={window})')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 8. Optimal Arm Selection Rate Over Time
    ax8 = plt.subplot(3, 3, 8)
    window = 1000
    eg_optimal = (eg_arms == optimal_arm).astype(int)
    ts_optimal = (ts_arms == optimal_arm).astype(int)
    eg_optimal_rate = np.convolve(eg_optimal, np.ones(window) / window, mode='valid')
    ts_optimal_rate = np.convolve(ts_optimal, np.ones(window) / window, mode='valid')
    ax8.plot(range(window - 1, num_trials), eg_optimal_rate, label='Epsilon-Greedy', alpha=0.8)
    ax8.plot(range(window - 1, num_trials), ts_optimal_rate, label='Thompson Sampling', alpha=0.8)
    ax8.axhline(y=1.0, color='r', linestyle='--', label='Perfect', alpha=0.5)
    ax8.set_xlabel('Trial')
    ax8.set_ylabel('Optimal Arm Selection Rate')
    ax8.set_title(f'Optimal Arm Selection Rate (window={window})')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # 9. Summary Statistics Table
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')

    eg_total_reward = np.sum(eg_rewards)
    ts_total_reward = np.sum(ts_rewards)
    eg_total_regret = optimal_reward * num_trials - eg_total_reward
    ts_total_regret = optimal_reward * num_trials - ts_total_reward
    eg_avg_reward = np.mean(eg_rewards)
    ts_avg_reward = np.mean(ts_rewards)
    eg_optimal_count = np.sum(eg_arms == optimal_arm)
    ts_optimal_count = np.sum(ts_arms == optimal_arm)

    stats_data = [
        ['Metric', 'Epsilon-Greedy', 'Thompson\nSampling', 'Winner'],
        ['Total Reward', f'{eg_total_reward:.0f}', f'{ts_total_reward:.0f}',
         'TS' if ts_total_reward > eg_total_reward else 'EG'],
        ['Avg Reward', f'{eg_avg_reward:.4f}', f'{ts_avg_reward:.4f}',
         'TS' if ts_avg_reward > eg_avg_reward else 'EG'],
        ['Total Regret', f'{eg_total_regret:.0f}', f'{ts_total_regret:.0f}',
         'EG' if eg_total_regret < ts_total_regret else 'TS'],
        ['Optimal Picks', f'{eg_optimal_count}', f'{ts_optimal_count}',
         'TS' if ts_optimal_count > eg_optimal_count else 'EG'],
        ['Optimal %', f'{eg_optimal_count / num_trials * 100:.2f}%',
         f'{ts_optimal_count / num_trials * 100:.2f}%',
         'TS' if ts_optimal_count > eg_optimal_count else 'EG']
    ]

    table = ax9.table(cellText=stats_data, cellLoc='center', loc='center',
                      colWidths=[0.30, 0.25, 0.25, 0.15],
                      bbox=[0, 0.05, 1, 0.85])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    for i in range(len(stats_data)):
        for j in range(len(stats_data[0])):
            cell = table[(i, j)]
            cell.set_edgecolor('gray')
            cell.set_linewidth(0.5)
            if i > 0:
                cell.set_facecolor('white' if i % 2 == 0 else '#f0f0f0')

    ax9.set_title('Performance Summary', fontsize=12, fontweight='bold', pad=5)

    plt.tight_layout(pad=2.5, h_pad=3.5, w_pad=2.5)
    plt.savefig('algorithm_comparison.png', dpi=300, bbox_inches='tight')
    logger.info("Comparison plot saved as 'algorithm_comparison.png'")
    plt.show()


if __name__ == '__main__':
    Bandit_Reward = [1, 2, 3, 4]

    logger.info("Starting Epsilon-Greedy Experiment...")
    epsilon_greedy = EpsilonGreedy(Bandit_Reward)

    eg_rewards, arms_selected_eg, num_times_explored, num_times_exploited, num_optimal = epsilon_greedy.experiment(
        num_trials=20000)

    epsilon_greedy.report(eg_rewards, arms_selected_eg, num_times_explored, num_times_exploited, num_trials=20000,
                          filename="epsilongreedy_results.csv")

    visualizer = Visualization()
    visualizer.plot1(eg_rewards, num_trials=20000)

    logger.info("Starting Thompson Sampling Experiment...")
    thompson_sampling = ThompsonSampling(Bandit_Reward, known_precision=10.0, prior_mean= 2.5, prior_variance=1)

    ts_rewards, arms_selected_ts, num_times_optimal = thompson_sampling.experiment(num_trials=20000)

    thompson_sampling.report(ts_rewards, arms_selected_ts, num_times_optimal, num_trials=20000,
                             filename="thompson_results.csv")

    visualizer.plot1(ts_rewards, num_trials=20000)
    visualizer.plot2(eg_rewards, ts_rewards, num_trials=20000, Bandit_Reward=Bandit_Reward)

    logger.info("Generating comprehensive comparison...")
    comparison(epsilon_greedy, thompson_sampling, eg_rewards, ts_rewards,
               arms_selected_eg, arms_selected_ts, num_trials=20000,
               Bandit_Reward=Bandit_Reward)

    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")
