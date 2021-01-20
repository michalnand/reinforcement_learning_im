import sys
sys.path.insert(0, '../../')

from libs_common.RLStatsCompute import *

import matplotlib.pyplot as plt

result_path = "./results/"

files = []
files.append("./models/dqn_baseline/result/result.log")
rl_stats_compute_dqn = RLStatsCompute(files)

files = []
files.append("./models/dqn_curiosity/result/result.log")
rl_stats_compute_curiosity = RLStatsCompute(files) 


plt.cla()
plt.ylabel("score")
plt.xlabel("episode")
plt.grid(color='black', linestyle='-', linewidth=0.1)

plt.plot(rl_stats_compute_dqn.mean[1], rl_stats_compute_dqn.mean[8], label="dqn baseline", color='deepskyblue')
plt.fill_between(rl_stats_compute_dqn.mean[1], rl_stats_compute_dqn.lower[8], rl_stats_compute_dqn.upper[8], color='deepskyblue', alpha=0.2)

plt.plot(rl_stats_compute_curiosity.mean[1], rl_stats_compute_curiosity.mean[8], label="dqn curiosity RND", color='limegreen')
plt.fill_between(rl_stats_compute_curiosity.mean[1], rl_stats_compute_curiosity.lower[8], rl_stats_compute_curiosity.upper[8], color='limegreen', alpha=0.2)

plt.legend(loc='lower right', borderaxespad=0.)
plt.savefig(result_path + "dqn_score_per_episode.png", dpi = 300)

 

plt.cla()
plt.ylabel("score")
plt.xlabel("iteration")
plt.grid(color='black', linestyle='-', linewidth=0.1)

plt.plot(rl_stats_compute_dqn.mean[0], rl_stats_compute_dqn.mean[8], label="dqn baseline", color='deepskyblue')
plt.fill_between(rl_stats_compute_dqn.mean[0], rl_stats_compute_dqn.lower[8], rl_stats_compute_dqn.upper[8], color='deepskyblue', alpha=0.2)

plt.plot(rl_stats_compute_curiosity.mean[0], rl_stats_compute_curiosity.mean[8], label="dqn curiosity RND", color='limegreen')
plt.fill_between(rl_stats_compute_curiosity.mean[0], rl_stats_compute_curiosity.lower[8], rl_stats_compute_curiosity.upper[8], color='limegreen', alpha=0.2)

plt.legend(loc='lower right', borderaxespad=0.)
plt.savefig(result_path + "dqn_score_per_iteration.png", dpi = 300)



plt.cla()
plt.ylabel("value")
plt.xlabel("iteration")
plt.grid(color='black', linestyle='-', linewidth=0.1)

plt.plot(rl_stats_compute_curiosity.mean[0], rl_stats_compute_curiosity.mean[10], label="curiosity", color='deepskyblue')
plt.fill_between(rl_stats_compute_curiosity.mean[0], rl_stats_compute_curiosity.lower[10], rl_stats_compute_curiosity.upper[10], color='deepskyblue', alpha=0.2)

plt.legend(loc='upper right', borderaxespad=0.)
plt.savefig(result_path + "dqn_curiosity_internal_motivation.png", dpi = 300)


'''
plt.cla()
plt.ylabel("value")
plt.xlabel("iteration")
plt.grid(color='black', linestyle='-', linewidth=0.1)

plt.plot(rl_stats_compute_entropy.mean[0], rl_stats_compute_entropy.mean[10], label="curiosity", color='deepskyblue')
plt.fill_between(rl_stats_compute_entropy.mean[0], rl_stats_compute_entropy.lower[10], rl_stats_compute_entropy.upper[10], color='deepskyblue', alpha=0.2)

plt.plot(rl_stats_compute_entropy.mean[0], rl_stats_compute_entropy.mean[12], label="entropy", color='red')
plt.fill_between(rl_stats_compute_entropy.mean[0], rl_stats_compute_entropy.lower[12], rl_stats_compute_entropy.upper[12], color='red', alpha=0.2)

plt.legend(loc='upper right', borderaxespad=0.)
plt.savefig(result_path + "dqn_entropy_internal_motivation.png", dpi = 300)

'''