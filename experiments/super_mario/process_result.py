import sys
sys.path.insert(0, '../../')

from libs_common.RLStatsCompute import *

import matplotlib.pyplot as plt

result_path = "./results/"

files = []
files.append("./models/dqn_baseline/result/result.log")
rl_stats_compute_dqn = RLStatsCompute(files, result_path + "dqn_baseline.log")

files = []
files.append("./models/dqn_curiosity/result/result.log")
rl_stats_compute_curiosity = RLStatsCompute(files, result_path + "dqn_curiosity.log") 

files = []
files.append("./models/dqn_curiosity_em/result/result.log")
rl_stats_compute_curiosity_em = RLStatsCompute(files, result_path + "dqn_curiosity_em.log") 



plt.cla()
plt.ylabel("score")
plt.xlabel("episode")
plt.grid(color='black', linestyle='-', linewidth=0.1)

plt.plot(rl_stats_compute_dqn.games_mean, rl_stats_compute_dqn.episode_mean, label="dqn baseline", color='deepskyblue')
plt.fill_between(rl_stats_compute_dqn.games_mean, rl_stats_compute_dqn.episode_lower, rl_stats_compute_dqn.episode_upper, color='deepskyblue', alpha=0.2)

plt.plot(rl_stats_compute_curiosity.games_mean, rl_stats_compute_curiosity.episode_mean, label="dqn curiosity", color='limegreen')
plt.fill_between(rl_stats_compute_curiosity.games_mean, rl_stats_compute_curiosity.episode_lower, rl_stats_compute_curiosity.episode_upper, color='limegreen', alpha=0.2)

plt.plot(rl_stats_compute_curiosity_em.games_mean, rl_stats_compute_curiosity_em.episode_mean, label="dqn curiosity em", color='red')
plt.fill_between(rl_stats_compute_curiosity_em.games_mean, rl_stats_compute_curiosity_em.episode_lower, rl_stats_compute_curiosity_em.episode_upper, color='red', alpha=0.2)

plt.legend(loc='lower right', borderaxespad=0.)
plt.savefig(result_path + "score_per_episode.png", dpi = 300)

 


plt.cla()
plt.ylabel("score")
plt.xlabel("iteration")
plt.grid(color='black', linestyle='-', linewidth=0.1)

plt.plot(rl_stats_compute_dqn.iterations, rl_stats_compute_dqn.episode_mean, label="dqn baseline", color='deepskyblue')
plt.fill_between(rl_stats_compute_dqn.iterations, rl_stats_compute_dqn.episode_lower, rl_stats_compute_dqn.episode_upper, color='deepskyblue', alpha=0.2)

plt.plot(rl_stats_compute_curiosity.iterations, rl_stats_compute_curiosity.episode_mean, label="dqn curiosity", color='limegreen')
plt.fill_between(rl_stats_compute_curiosity.iterations, rl_stats_compute_curiosity.episode_lower, rl_stats_compute_curiosity.episode_upper, color='limegreen', alpha=0.2)

plt.plot(rl_stats_compute_curiosity_em.iterations, rl_stats_compute_curiosity_em.episode_mean, label="dqn curiosity em", color='red')
plt.fill_between(rl_stats_compute_curiosity_em.iterations, rl_stats_compute_curiosity_em.episode_lower, rl_stats_compute_curiosity_em.episode_upper, color='red', alpha=0.2)

plt.legend(loc='lower right', borderaxespad=0.)
plt.savefig(result_path + "score_per_iteration.png", dpi = 300)


'''
plt.cla()
plt.ylabel("value")
plt.xlabel("iteration")
plt.grid(color='black', linestyle='-', linewidth=0.1)

plt.plot(rl_stats_compute_curiosity.iterations, rl_stats_compute_curiosity.curiosity_mean, label="curiosity", color='deepskyblue')
plt.fill_between(rl_stats_compute_curiosity.iterations, rl_stats_compute_curiosity.curiosity_lower, rl_stats_compute_curiosity.curiosity_upper, color='deepskyblue', alpha=0.2)

#plt.plot(rl_stats_compute_curiosity_em.iterations, rl_stats_compute_curiosity_em.curiosity_mean, label="curiosity entropy", color='red')
#plt.fill_between(rl_stats_compute_curiosity_em.iterations, rl_stats_compute_curiosity_em.curiosity_lower, rl_stats_compute_curiosity_em.curiosity_upper, color='red', alpha=0.2)

plt.legend(loc='lower right', borderaxespad=0.)
plt.savefig(result_path + "internal_motivation.png", dpi = 300)
'''


'''
plt.cla()
plt.ylabel("value")
plt.xlabel("iteration")
plt.grid(color='black', linestyle='-', linewidth=0.1)

plt.plot(rl_stats_compute_curiosity.iterations, rl_stats_compute_curiosity.forward_loss_mean, label="curiosity", color='deepskyblue')
plt.fill_between(rl_stats_compute_curiosity.iterations, rl_stats_compute_curiosity.forward_loss_lower, rl_stats_compute_curiosity.forward_loss_upper, color='deepskyblue', alpha=0.2)

plt.plot(rl_stats_compute_curiosity_em.iterations, rl_stats_compute_curiosity_em.forward_loss_mean, label="curiosity entropy", color='deepskyblue')
plt.fill_between(rl_stats_compute_curiosity_em.iterations, rl_stats_compute_curiosity_em.forward_loss_lower, rl_stats_compute_curiosity_em.forward_loss_upper, color='deepskyblue', alpha=0.2)

plt.legend(loc='lower right', borderaxespad=0.)
plt.savefig(result_path + "forward_model_loss.png", dpi = 300)
'''