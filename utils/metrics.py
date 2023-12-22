import numpy as np
import time, datetime
import matplotlib.pyplot as plt

# Metric logger for Double Deep Q Network (DDQN)
class DDQNMetricLogger():
    def __init__(self, save_dir):
        # Initialize log files and plots
        self.save_log = save_dir / "log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"

        # History metrics
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        # Moving averages, added for every call to record()
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        # Current episode metric
        self.init_episode()

        # Timing
        self.record_time = time.time()

    # Log a step in the training process
    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    # Log the end of an episode
    def log_episode(self):
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

        self.init_episode()

    # Initialize current episode metric
    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    # Record metrics for the current episode
    def record(self, episode, epsilon, step):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Epsilon {epsilon} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean Loss {mean_ep_loss} - "
            f"Mean Q Value {mean_ep_q} - "
            f"Time Delta {time_since_last_record} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["ep_rewards", "ep_lengths", "ep_avg_losses", "ep_avg_qs"]:
            plt.plot(getattr(self, f"moving_avg_{metric}"))
            plt.savefig(getattr(self, f"{metric}_plot"))
            plt.clf()

# Metric logger for Asynchronous Advantage Actor-Critic (A3C)
class A3CMetricLogger():
    def __init__(self, save_dir):
        self.save_log = save_dir + "/log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanActorLoss':>15}{'MeancCiticLoss':>15}{'MeanEntropyLoss':>15}{'MeanTotalLoss':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
        self.ep_rewards_plot = save_dir + "/reward_plot.jpg"
        self.ep_lengths_plot = save_dir + "/length_plot.jpg"
        self.ep_avg_actor_losses_plot = save_dir + "/actor_loss_plot.jpg"
        self.ep_avg_critic_losses_plot = save_dir + "/critic_loss_plot.jpg"
        self.ep_avg_entropy_losses_plot = save_dir + "/entropy_loss_plot.jpg"
        self.ep_avg_total_losses_plot = save_dir + "/total_loss_plot.jpg"

        # History metrics
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_actor_losses = []
        self.ep_avg_critic_losses = []
        self.ep_avg_entropy_losses = []
        self.ep_avg_total_losses = []

        # Moving averages, added for every call to record()
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_actor_losses = []
        self.moving_avg_ep_avg_critic_losses = []
        self.moving_avg_ep_avg_entropy_losses = []
        self.moving_avg_ep_avg_total_losses = []
        
        # Current episode metric
        self.init_episode()

        # Timing
        self.record_time = time.time()


    def log_step(self, reward, actor_loss, critic_loss, entropy_loss, total_loss):
        if not reward == None:
            self.curr_ep_reward += reward
            self.curr_ep_length += 1
        if not any([loss == None for loss in [actor_loss, critic_loss, entropy_loss, total_loss]]):
            self.curr_ep_actor_loss += actor_loss.item()
            self.curr_ep_critic_loss += critic_loss.item()
            self.curr_ep_entropy_loss += entropy_loss.item()
            self.curr_ep_total_loss += total_loss.item()
            self.curr_ep_loss_length += 1

    def log_episode(self):
        "Mark end of episode"
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_actor_loss = 0
            ep_avg_critic_loss = 0
            ep_avg_entropy_loss = 0
            ep_avg_total_loss = 0
        else:
            ep_avg_actor_loss = np.round(self.curr_ep_actor_loss / self.curr_ep_loss_length, 5)
            ep_avg_critic_loss = np.round(self.curr_ep_critic_loss / self.curr_ep_loss_length, 5)
            ep_avg_entropy_loss = np.round(self.curr_ep_entropy_loss / self.curr_ep_loss_length, 5)
            ep_avg_total_loss  = np.round(self.curr_ep_loss_length / self.curr_ep_loss_length, 5)
        self.ep_avg_actor_losses.append(ep_avg_actor_loss)
        self.ep_avg_critic_losses.append(ep_avg_critic_loss)
        self.ep_avg_entropy_losses.append(ep_avg_entropy_loss)
        self.ep_avg_total_losses.append(ep_avg_total_loss)
        
        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_actor_loss = 0
        self.curr_ep_critic_loss = 0
        self.curr_ep_entropy_loss = 0
        self.curr_ep_total_loss = 0
        self.curr_ep_loss_length = 0

    def record(self, episode):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        
        mean_ep_actor_loss = np.round(np.mean(self.ep_avg_actor_losses[-100:]), 3)
        mean_ep_critic_loss = np.round(np.mean(self.ep_avg_critic_losses[-100:]), 3)
        mean_ep_entropy_loss = np.round(np.mean(self.ep_avg_entropy_losses[-100:]), 3)
        mean_ep_total_loss = np.round(np.mean(self.ep_avg_total_losses[-100:]), 3)

        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_actor_losses.append(mean_ep_actor_loss)
        self.moving_avg_ep_avg_critic_losses.append(mean_ep_critic_loss)
        self.moving_avg_ep_avg_entropy_losses.append(mean_ep_entropy_loss)
        self.moving_avg_ep_avg_total_losses.append(mean_ep_total_loss)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean Actor Loss {mean_ep_actor_loss} - "
            f"Mean Critic Loss {mean_ep_critic_loss} - "
            f"Mean Entropy Loss {mean_ep_entropy_loss} - "
            f"Mean Total Loss {mean_ep_total_loss} - "
            f"Time Delta {time_since_last_record} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_actor_loss:15.3f}{mean_ep_critic_loss:15.3f}{mean_ep_entropy_loss:15.3f}{mean_ep_total_loss:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["ep_rewards", "ep_lengths", "ep_avg_actor_losses", "ep_avg_critic_losses", "ep_avg_entropy_losses", "ep_avg_total_losses"]:
            plt.plot(getattr(self, f"moving_avg_{metric}"))
            plt.savefig(getattr(self, f"{metric}_plot"))
            plt.clf()

# Metric logger for Proximal Policy Optimization (PPO)
class PPOMetricLogger():
    def __init__(self, save_dir):
        self.save_log = save_dir + "/log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanActorLoss':>15}{'MeancCiticLoss':>15}{'MeanEntropyLoss':>15}{'MeanTotalLoss':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
        self.ep_rewards_plot = save_dir + "/reward_plot.jpg"
        self.ep_lengths_plot = save_dir + "/length_plot.jpg"
        self.ep_avg_actor_losses_plot = save_dir + "/actor_loss_plot.jpg"
        self.ep_avg_critic_losses_plot = save_dir + "/critic_loss_plot.jpg"
        self.ep_avg_entropy_losses_plot = save_dir + "/entropy_loss_plot.jpg"
        self.ep_avg_total_losses_plot = save_dir + "/total_loss_plot.jpg"

        # History metrics
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_actor_losses = []
        self.ep_avg_critic_losses = []
        self.ep_avg_entropy_losses = []
        self.ep_avg_total_losses = []

        # Moving averages, added for every call to record()
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_actor_losses = []
        self.moving_avg_ep_avg_critic_losses = []
        self.moving_avg_ep_avg_entropy_losses = []
        self.moving_avg_ep_avg_total_losses = []
        
        # Current episode metric
        self.init_episode()

        # Timing
        self.record_time = time.time()


    def log_step(self, reward, actor_loss, critic_loss, entropy_loss, total_loss):
        if not reward == None:
            self.curr_ep_reward += reward.cpu().numpy().mean()
            self.curr_ep_length += 1
        if not any([loss == None for loss in [actor_loss, critic_loss, entropy_loss, total_loss]]):
            self.curr_ep_actor_loss += actor_loss.cpu().item()
            self.curr_ep_critic_loss += critic_loss.cpu().item()
            self.curr_ep_entropy_loss += entropy_loss.cpu().item()
            self.curr_ep_total_loss += total_loss.cpu().item()
            self.curr_ep_loss_length += 1

    def log_episode(self):
        "Mark end of episode"
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_actor_loss = 0
            ep_avg_critic_loss = 0
            ep_avg_entropy_loss = 0
            ep_avg_total_loss = 0
        else:
            ep_avg_actor_loss = np.round(self.curr_ep_actor_loss / self.curr_ep_loss_length, 5)
            ep_avg_critic_loss = np.round(self.curr_ep_critic_loss / self.curr_ep_loss_length, 5)
            ep_avg_entropy_loss = np.round(self.curr_ep_entropy_loss / self.curr_ep_loss_length, 5)
            ep_avg_total_loss  = np.round(self.curr_ep_loss_length / self.curr_ep_loss_length, 5)
                
        self.ep_avg_actor_losses.append(ep_avg_actor_loss)
        self.ep_avg_critic_losses.append(ep_avg_critic_loss)
        self.ep_avg_entropy_losses.append(ep_avg_entropy_loss)
        self.ep_avg_total_losses.append(ep_avg_total_loss)

        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_actor_loss = 0
        self.curr_ep_critic_loss = 0
        self.curr_ep_entropy_loss = 0
        self.curr_ep_total_loss = 0
        self.curr_ep_loss_length = 0

    def record(self, episode):

        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        
        mean_ep_actor_loss = np.round(np.mean(self.ep_avg_actor_losses[-100:]), 3)
        mean_ep_critic_loss = np.round(np.mean(self.ep_avg_critic_losses[-100:]), 3)
        mean_ep_entropy_loss = np.round(np.mean(self.ep_avg_entropy_losses[-100:]), 3)
        mean_ep_total_loss = np.round(np.mean(self.ep_avg_total_losses[-100:]), 3)


        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_actor_losses.append(mean_ep_actor_loss)
        self.moving_avg_ep_avg_critic_losses.append(mean_ep_critic_loss)
        self.moving_avg_ep_avg_entropy_losses.append(mean_ep_entropy_loss)
        self.moving_avg_ep_avg_total_losses.append(mean_ep_total_loss)


        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean Actor Loss {mean_ep_actor_loss} - "
            f"Mean Critic Loss {mean_ep_critic_loss} - "
            f"Mean Entropy Loss {mean_ep_entropy_loss} - "
            f"Mean Total Loss {mean_ep_total_loss} - "
            f"Time Delta {time_since_last_record} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_actor_loss:15.3f}{mean_ep_critic_loss:15.3f}{mean_ep_entropy_loss:15.3f}{mean_ep_total_loss:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["ep_rewards", "ep_lengths", "ep_avg_actor_losses", "ep_avg_critic_losses", "ep_avg_entropy_losses", "ep_avg_total_losses"]:
            plt.plot(getattr(self, f"moving_avg_{metric}"))
            plt.savefig(getattr(self, f"{metric}_plot"))
            plt.clf()