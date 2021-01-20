import numpy

import matplotlib.pyplot as plt


class RoomsEnv:

    def __init__(self, size, room_size = 8):
        self.size      = size
        self.room_size = room_size

    def reset(self):
        self.rooms      = numpy.zeros((self.size, self.room_size))
        self.visited    = numpy.zeros(self.size, dtype=bool)

        self.position   = self.room_size-1

        self.steps = 0

        for r in range(self.size):
            self.rooms[r] = numpy.random.randn(1)*numpy.ones(self.room_size) + 0.1*numpy.random.randn(self.room_size)

        return self._upadate_observation()


    def step(self, action):
        
        room_idx    = self.position//self.room_size
        self.visited[room_idx] = True

        if action == 0:
            self.position+= 1
        elif action == 1:
            self.position-= 1
        else:
            pass

        if self.position < 0:
            self.position = 0
        if self.position > self.size*self.room_size-1:
            self.position = self.size*self.room_size-1


        reward = 0
        done   = False

        self.steps+= 1

        if numpy.all(self.visited):
            reward = 1.0
            done   = True
        elif self.steps > 2*self.size*self.room_size:
            reward = -1.0
            done   = True
        

        return self._upadate_observation(), reward, done, None
        


    def _upadate_observation(self):
        room_idx    = self.position//self.room_size
        player_idx  = self.position%self.room_size
        
        result      = self.rooms[room_idx].copy()

        result[player_idx] = 1.0

        return result


def action_one_hot(action, count):
    result = numpy.zeros(count)
    result[action] = 1.0

    return result

def kl_divergence(mu1, sigma1, mu2, sigma2):
    result = numpy.log(sigma2/sigma1)
    result+= (sigma1**2 + (mu1 - mu2)**2)/(2.0*(sigma2**2))
    result+= -0.5

    return result

def compute_motivation(env, policy, steps = 20000, beta=0.01):
    result = []

    obs             = env.reset()
    
    em_size         = 256

    episodic_memory_obs = numpy.zeros( (em_size, ) + obs.shape )
    for i in range(em_size):
        episodic_memory_obs[i] = obs.copy()

    episodic_memory_actions = numpy.zeros((em_size, 3))


    score = 0

    em_idx = 0
    
    for s in range(steps):
        action = policy()
        obs, reward, done, _ = env.step(action)

        action_one_hot_ = action_one_hot(action, 3)

        distance_obs     = ((episodic_memory_obs - obs)**2).mean(axis=1)
        distance_actions = ((episodic_memory_actions - action_one_hot_)**2).mean(axis=1)

        
        episodic_memory_obs[em_idx]             = obs
        episodic_memory_actions[em_idx]         = action_one_hot_

        #motivation = numpy.tanh(beta*distance_obs.mean())
        #motivation = numpy.tanh(beta*distance_obs.std())

        #motivation = numpy.tanh(beta*distance_obs.mean()/(0.01 + distance_actions.mean()))
        #motivation = numpy.tanh(beta*distance_obs.std()/(0.01 + distance_actions.std()))

        episodic_memory_obs_std      = episodic_memory_obs.std(axis=0).mean()        
        episodic_memory_actions_std  = episodic_memory_actions.std(axis=0).mean()
        
        motivation                 = beta*numpy.tanh(episodic_memory_obs_std)
        #motivation                = beta*numpy.tanh(episodic_memory_obs_std/(0.01 + episodic_memory_actions_std))

        result.append(motivation)

        if done:
            obs = env.reset()
            for i in range(em_size):
                episodic_memory_obs[i] = obs.copy()
            
            episodic_memory_action = numpy.zeros((em_size, 3))

        em_idx = (em_idx + 1)%em_size
        score+= reward
    
    return numpy.array(result), score

def policy_random():
    return numpy.random.randint(3)


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

def policy_optimal():
    if numpy.random.randint(100) < 5:
        return policy_random()
    else:
        return 0

if __name__ == "__main__":
    env = RoomsEnv(10000, 1000)

    motivation_random, score_random  = compute_motivation(env, policy_random)
    motivation_procast, score_procast = compute_motivation(env, policy_procast)
    motivation_optimal, score_optimal = compute_motivation(env, policy_optimal)

    print(motivation_random.mean(), motivation_random.std(), score_random)
    print(motivation_procast.mean(), motivation_procast.std(), score_procast)
    print(motivation_optimal.mean(), motivation_optimal.std(), score_optimal)
    
    plt.hist(motivation_random,  bins = 64, label="random " + str(numpy.round(motivation_random.mean(), 5)), alpha=0.7)
    plt.hist(motivation_procast, bins = 64, label="procastinating " + str(numpy.round(motivation_procast.mean(), 5)), alpha=0.7)
    plt.hist(motivation_optimal, bins = 64, label="optimal " + str(numpy.round(motivation_optimal.mean(), 5)), alpha=0.7)
    plt.legend()
    plt.show()