import sys
sys.path.insert(0, '../../')

from libs_common.RLStatsCompute import *

import matplotlib.pyplot as plt

result_path = "./results/"

files = []
files.append("./models/ddpg_baseline/run_0/result/result.log")
files.append("./models/ddpg_baseline/run_1/result/result.log")
files.append("./models/ddpg_baseline/run_2/result/result.log")
files.append("./models/ddpg_baseline/run_3/result/result.log")
files.append("./models/ddpg_baseline/run_4/result/result.log")
files.append("./models/ddpg_baseline/run_5/result/result.log")
files.append("./models/ddpg_baseline/run_6/result/result.log")
files.append("./models/ddpg_baseline/run_7/result/result.log")
rl_stats_compute_ddpg = RLStatsCompute(files, result_path + "ddpg_baseline.log")

files = []
files.append("./models/ddpg_curiosity/run_0/result/result.log")
files.append("./models/ddpg_curiosity/run_1/result/result.log")
files.append("./models/ddpg_curiosity/run_2/result/result.log")
files.append("./models/ddpg_curiosity/run_3/result/result.log")
files.append("./models/ddpg_curiosity/run_4/result/result.log")
files.append("./models/ddpg_curiosity/run_5/result/result.log")
files.append("./models/ddpg_curiosity/run_6/result/result.log")
files.append("./models/ddpg_curiosity/run_7/result/result.log")
rl_stats_compute_curiosity = RLStatsCompute(files, result_path + "ddpg_curiosity.log") 


files = []
files.append("./models/ddpg_curiosity_em/model/result/result.log")
rl_stats_compute_curiosity_em = RLStatsCompute(files, result_path + "ddpg_curiosity_em.log") 


plt.cla()
plt.ylabel("score")
plt.xlabel("episode")
plt.grid(color='black', linestyle='-', linewidth=0.1)

plt.plot(rl_stats_compute_ddpg.games_mean, rl_stats_compute_ddpg.episode_mean, label="ddpg baseline", color='blue')
plt.fill_between(rl_stats_compute_ddpg.games_mean, rl_stats_compute_ddpg.episode_lower, rl_stats_compute_ddpg.episode_upper, color='blue', alpha=0.2)


plt.plot(rl_stats_compute_curiosity.games_mean, rl_stats_compute_curiosity.episode_mean, label="ddpg curiosity", color='red')
plt.fill_between(rl_stats_compute_curiosity.games_mean, rl_stats_compute_curiosity.episode_lower, rl_stats_compute_curiosity.episode_upper, color='red', alpha=0.2)

plt.plot(rl_stats_compute_curiosity_em.games_mean, rl_stats_compute_curiosity_em.episode_mean, label="ddpg curiosity em", color='green')
plt.fill_between(rl_stats_compute_curiosity_em.games_mean, rl_stats_compute_curiosity_em.episode_lower, rl_stats_compute_curiosity_em.episode_upper, color='green', alpha=0.2)

plt.legend(loc='lower right', borderaxespad=0.)
plt.savefig(result_path + "score_per_episode.png", dpi = 300)

 


plt.cla()
plt.ylabel("score")
plt.xlabel("iteration")
plt.grid(color='black', linestyle='-', linewidth=0.1)

plt.plot(rl_stats_compute_ddpg.iterations, rl_stats_compute_ddpg.episode_mean, label="ddpg baseline", color='blue')
plt.fill_between(rl_stats_compute_ddpg.iterations, rl_stats_compute_ddpg.episode_lower, rl_stats_compute_ddpg.episode_upper, color='blue', alpha=0.2)

plt.plot(rl_stats_compute_curiosity.iterations, rl_stats_compute_curiosity.episode_mean, label="ddpg curiosity", color='red')
plt.fill_between(rl_stats_compute_curiosity.iterations, rl_stats_compute_curiosity.episode_lower, rl_stats_compute_curiosity.episode_upper, color='red', alpha=0.2)

plt.plot(rl_stats_compute_curiosity_em.iterations, rl_stats_compute_curiosity_em.episode_mean, label="ddpg curiosity em", color='green')
plt.fill_between(rl_stats_compute_curiosity_em.iterations, rl_stats_compute_curiosity_em.episode_lower, rl_stats_compute_curiosity_em.episode_upper, color='green', alpha=0.2)

plt.legend(loc='lower right', borderaxespad=0.)
plt.savefig(result_path + "score_per_iteration.png", dpi = 300)


'''

plt.cla()
plt.ylabel("value")
plt.xlabel("iteration")
plt.grid(color='black', linestyle='-', linewidth=0.1)

plt.plot(rl_stats_compute_curiosity.iterations, rl_stats_compute_curiosity.entropy_mean, label="entropy", color='orange')
plt.fill_between(rl_stats_compute_curiosity.iterations, rl_stats_compute_curiosity.entropy_lower, rl_stats_compute_curiosity.entropy_upper, color='orange', alpha=0.2)


plt.plot(rl_stats_compute_curiosity.iterations, rl_stats_compute_curiosity.curiosity_mean, label="curiosity", color='green')
plt.fill_between(rl_stats_compute_curiosity.iterations, rl_stats_compute_curiosity.curiosity_lower, rl_stats_compute_curiosity.curiosity_upper, color='green', alpha=0.2)

plt.legend(loc='lower right', borderaxespad=0.)
plt.savefig(result_path + "internal_motivation.png", dpi = 300)



plt.cla()
plt.ylabel("value")
plt.xlabel("iteration")
plt.grid(color='black', linestyle='-', linewidth=0.1)

plt.plot(rl_stats_compute_curiosity.iterations, rl_stats_compute_curiosity.forward_loss_mean, label="forward model loss", color='navy')
plt.fill_between(rl_stats_compute_curiosity.iterations, rl_stats_compute_curiosity.forward_loss_lower, rl_stats_compute_curiosity.forward_loss_upper, color='navy', alpha=0.2)

plt.legend(loc='lower right', borderaxespad=0.)
plt.savefig(result_path + "forward_model_loss.png", dpi = 300)
'''