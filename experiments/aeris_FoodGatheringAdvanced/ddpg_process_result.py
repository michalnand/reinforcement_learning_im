import sys
sys.path.insert(0, '../../')

import RLAgents

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
ddpg_baseline = RLAgents.RLStatsCompute(files)

files = []
files.append("./models/ddpg_curiosity/run_0/result/result.log")
files.append("./models/ddpg_curiosity/run_1/result/result.log")
files.append("./models/ddpg_curiosity/run_2/result/result.log")
files.append("./models/ddpg_curiosity/run_3/result/result.log")
files.append("./models/ddpg_curiosity/run_4/result/result.log")
files.append("./models/ddpg_curiosity/run_5/result/result.log")
files.append("./models/ddpg_curiosity/run_6/result/result.log")
files.append("./models/ddpg_curiosity/run_7/result/result.log")
ddpg_curiosity_rnd = RLAgents.RLStatsCompute(files) 

'''
files = []
files.append("./models/ddpg_entropy/run_0/result/result.log")
files.append("./models/ddpg_entropy/run_1/result/result.log")
files.append("./models/ddpg_entropy/run_2/result/result.log")
files.append("./models/ddpg_entropy/run_3/result/result.log")
files.append("./models/ddpg_entropy/run_4/result/result.log")
files.append("./models/ddpg_entropy/run_5/result/result.log")
files.append("./models/ddpg_entropy/run_6/result/result.log")
files.append("./models/ddpg_entropy/run_7/result/result.log")
ddpg_curiosity_entropy = RLAgents.RLStatsCompute(files) 
'''

plt.cla()
plt.ylabel("score")
plt.xlabel("episode")
plt.grid(color='black', linestyle='-', linewidth=0.1)

plt.plot(ddpg_baseline.mean[1], ddpg_baseline.mean[4], label="ddpg baseline", color='deepskyblue')
plt.fill_between(ddpg_baseline.mean[1], ddpg_baseline.lower[4], ddpg_baseline.upper[4], color='deepskyblue', alpha=0.2)

plt.plot(ddpg_curiosity_rnd.mean[1], ddpg_curiosity_rnd.mean[4], label="ddpg curiosity RND", color='limegreen')
plt.fill_between(ddpg_curiosity_rnd.mean[1], ddpg_curiosity_rnd.lower[4], ddpg_curiosity_rnd.upper[4], color='limegreen', alpha=0.2)

#plt.plot(ddpg_curiosity_entropy.mean[1], ddpg_curiosity_entropy.mean[4], label="ddpg curiosity RND + entropy", color='red')
#plt.fill_between(ddpg_curiosity_entropy.mean[1], ddpg_curiosity_entropy.lower[4], ddpg_curiosity_entropy.upper[4], color='red', alpha=0.2)

plt.legend(loc='lower right', borderaxespad=0.)
plt.savefig(result_path + "ddpg_score_per_episode.png", dpi = 300)

 

plt.cla()
plt.ylabel("score")
plt.xlabel("iteration")
plt.grid(color='black', linestyle='-', linewidth=0.1)

plt.plot(ddpg_baseline.mean[0], ddpg_baseline.mean[4], label="ddpg baseline", color='deepskyblue')
plt.fill_between(ddpg_baseline.mean[0], ddpg_baseline.lower[4], ddpg_baseline.upper[4], color='deepskyblue', alpha=0.2)

plt.plot(ddpg_curiosity_rnd.mean[0], ddpg_curiosity_rnd.mean[4], label="ddpg curiosity RND", color='limegreen')
plt.fill_between(ddpg_curiosity_rnd.mean[0], ddpg_curiosity_rnd.lower[4], ddpg_curiosity_rnd.upper[4], color='limegreen', alpha=0.2)

#plt.plot(ddpg_curiosity_entropy.mean[0], ddpg_curiosity_entropy.mean[4], label="ddpg curiosity RND + entropy", color='red')
#plt.fill_between(ddpg_curiosity_entropy.mean[0], ddpg_curiosity_entropy.lower[4], ddpg_curiosity_entropy.upper[4], color='red', alpha=0.2)


plt.legend(loc='lower right', borderaxespad=0.)
plt.savefig(result_path + "ddpg_score_per_iteration.png", dpi = 300)







plt.cla()
plt.ylabel("value")
plt.xlabel("iteration")
plt.grid(color='black', linestyle='-', linewidth=0.1)

plt.plot(ddpg_curiosity_rnd.mean[0], ddpg_curiosity_rnd.mean[10], label="curiosity", color='deepskyblue')
plt.fill_between(ddpg_curiosity_rnd.mean[0], ddpg_curiosity_rnd.lower[10], ddpg_curiosity_rnd.upper[10], color='deepskyblue', alpha=0.2)

plt.legend(loc='upper right', borderaxespad=0.)
plt.savefig(result_path + "ddpg_curiosity_internal_motivation.png", dpi = 300)


'''
plt.cla()
plt.ylabel("value")
plt.xlabel("iteration")
plt.grid(color='black', linestyle='-', linewidth=0.1)

plt.plot(ddpg_curiosity_entropy.mean[0], ddpg_curiosity_entropy.mean[10], label="curiosity", color='deepskyblue')
plt.fill_between(ddpg_curiosity_entropy.mean[0], ddpg_curiosity_entropy.lower[10], ddpg_curiosity_entropy.upper[10], color='deepskyblue', alpha=0.2)

plt.plot(ddpg_curiosity_entropy.mean[0], ddpg_curiosity_entropy.mean[12], label="entropy", color='red')
plt.fill_between(ddpg_curiosity_entropy.mean[0], ddpg_curiosity_entropy.lower[12], ddpg_curiosity_entropy.upper[12], color='red', alpha=0.2)

plt.legend(loc='upper right', borderaxespad=0.)
plt.savefig(result_path + "ddpg_entropy_internal_motivation.png", dpi = 300)





plt.cla()
plt.ylabel("value")
plt.xlabel("iteration")
plt.grid(color='black', linestyle='-', linewidth=0.1)

plt.plot(ddpg_curiosity_entropy.mean[0], ddpg_curiosity_entropy.mean[9], label="RND loss", color='deepskyblue')
plt.fill_between(ddpg_curiosity_entropy.mean[0], ddpg_curiosity_entropy.lower[9], ddpg_curiosity_entropy.upper[9], color='deepskyblue', alpha=0.2)

plt.plot(ddpg_curiosity_entropy.mean[0], ddpg_curiosity_entropy.mean[11], label="AE loss", color='red')
plt.fill_between(ddpg_curiosity_entropy.mean[0], ddpg_curiosity_entropy.lower[11], ddpg_curiosity_entropy.upper[11], color='red', alpha=0.2)

plt.legend(loc='upper right', borderaxespad=0.)
plt.savefig(result_path + "ddpg_entropy_loss.png", dpi = 300)
'''