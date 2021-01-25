import numpy
import gym
import gym_explore

import matplotlib.pyplot as plt



def compute_motivation(env, policy, steps = 1000, em_size = 64):
    result = []

    obs             = env.reset()
    
    episodic_memory_obs = numpy.zeros( (em_size, obs.flatten().shape[0]) )
    for i in range(em_size):
        episodic_memory_obs[i] = obs.flatten()

    em_idx = 0
    
    for s in range(steps):

        action = policy()
        obs, reward, done, _ = env.step(action)

        std_prev                    = episodic_memory_obs.std(axis=0).mean()
        episodic_memory_obs[em_idx] = obs.flatten()
        std_now                     = episodic_memory_obs.std(axis=0).mean()


        motivation = numpy.tanh(std_now - std_prev)
        result.append(motivation)

        if done:
            obs = env.reset()
            for i in range(em_size):
                episodic_memory_obs[i] = obs.flatten()
            
        em_idx = (em_idx + 1)%em_size
    
    return numpy.array(result)

def policy_random():
    return numpy.random.randint(5)


policy_procast_state = 0

def policy_procast():
    global policy_procast_state

    if policy_procast_state == 0:
        action = 0
        policy_procast_state = 1
    else:
        action = 1
        policy_procast_state = 0

    return action

action_optimal = 0
def policy_optimal():
    global action_optimal
    if numpy.random.randint(100) < 20:
        action_optimal = policy_random()
    
    return action_optimal

if __name__ == "__main__":
    env = gym.make("ExploreArcadeGeneric-v0", size=16)

    motivation_random  = compute_motivation(env, policy_random)
    motivation_procast = compute_motivation(env, policy_procast)
    motivation_optimal = compute_motivation(env, policy_optimal)

    print(motivation_random.mean(), motivation_random.std())
    print(motivation_procast.mean(), motivation_procast.std())
    print(motivation_optimal.mean(), motivation_optimal.std())
    
    plt.hist(motivation_random,  bins = 64, label="random " + str(numpy.round(motivation_random.mean(), 5)), alpha=0.7)
    plt.hist(motivation_procast, bins = 64, label="procastinating " + str(numpy.round(motivation_procast.mean(), 5)), alpha=0.7)
    plt.hist(motivation_optimal, bins = 64, label="optimal " + str(numpy.round(motivation_optimal.mean(), 5)), alpha=0.7)
    plt.legend()
    plt.show()