import torch
import numpy

class PolicyBuffer:

    def __init__(self, buffer_size, state_shape, actions_size, envs_count, device):
        
        self.buffer_size    = buffer_size
        self.state_shape    = state_shape
        self.actions_size   = actions_size
        self.envs_count     = envs_count
        self.device         = device

        self.clear()

    def add(self, env, state, logits, value, action, reward, done):
        
        self.states_b[env][self.ptr]    = state
        self.logits_b[env][self.ptr]    = logits
        self.values_b[env][self.ptr]    = value
        self.actions_b[env][self.ptr]   = action
        self.rewards_b[env][self.ptr]   = reward
        self.dones_b[env][self.ptr]     = done
        
        if env == self.envs_count - 1:
            self.ptr = self.ptr + 1 


    def is_full(self):
        if self.ptr >= self.buffer_size:
            return True

        return False 
 
    def clear(self):
        self.states_b           = torch.zeros((self.envs_count, self.buffer_size, ) + self.state_shape, dtype=torch.float).to(self.device)
        self.logits_b           = torch.zeros((self.envs_count, self.buffer_size, self.actions_size), dtype=torch.float).to(self.device)
        
        self.values_b           = torch.zeros((self.envs_count, self.buffer_size, ), dtype=torch.float).to(self.device)

        self.actions_b          = torch.zeros((self.envs_count, self.buffer_size, ), dtype=int).to(self.device)
        self.rewards_b          = torch.zeros((self.envs_count, self.buffer_size, ), dtype=torch.float).to(self.device)
        self.dones_b            = torch.zeros((self.envs_count, self.buffer_size, ), dtype=torch.float).to(self.device)
       
        self.returns_b          = torch.zeros((self.envs_count, self.buffer_size, ), dtype=torch.float).to(self.device)
        self.advantages_b       = torch.zeros((self.envs_count, self.buffer_size, ), dtype=torch.float).to(self.device)

        self.ptr = 0 


    def compute_gae_returns(self, gamma, lam = 0.95):
        for e in range(self.envs_count):
            gae  = 0.0
            for n in reversed(range(len(self.rewards_b[e])-1)):
                if self.dones_b[e][n] > 0:
                    gamma_ = 0.0
                else:
                    gamma_ = gamma

                delta = self.rewards_b[e][n] + gamma_*self.values_b[e][n+1] - self.values_b[e][n]

                gae = delta + lam*gamma_*gae

                self.returns_b[e][n] = gae + self.values_b[e][n]

            self.advantages_b[e] = self.returns_b[e] - self.values_b[e]
            self.advantages_b[e] = (self.advantages_b[e] - torch.mean(self.advantages_b[e]))/(torch.std(self.advantages_b[e]) + 1e-10)


    def sample_batch(self, batch_size):

        states           = torch.zeros((batch_size, ) + self.state_shape, dtype=torch.float).to(self.device)
        logits           = torch.zeros((batch_size, self.actions_size), dtype=torch.float).to(self.device)
        
        values           = torch.zeros((batch_size, ), dtype=torch.float).to(self.device)

        actions          = torch.zeros((batch_size, ), dtype=int).to(self.device)
        rewards          = torch.zeros((batch_size, ), dtype=torch.float).to(self.device)
        dones            = torch.zeros((batch_size, ), dtype=torch.float).to(self.device)
       
        returns          = torch.zeros((batch_size, ), dtype=torch.float).to(self.device)
        advantages       = torch.zeros((batch_size, ), dtype=torch.float).to(self.device)

        for i in range(batch_size):
            env = numpy.random.randint(self.envs_count)
            n   = numpy.random.randint(self.buffer_size)

            states[i] = self.states_b[env][n]
            logits[i] = self.logits_b[env][n]
            
            values[i] = self.values_b[env][n]
            
            actions[i]      = self.actions_b[env][n]
            rewards[i]      = self.rewards_b[env][n]
            dones[i]        = self.dones_b[env][n]

            returns[i]      = self.returns_b[env][n]
            advantages[i]   = self.advantages_b[env][n]
       
        return states, logits, values, actions, rewards, dones, returns, advantages 