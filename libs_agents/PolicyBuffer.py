import torch
import numpy


class PolicyBuffer:

    def __init__(self, buffer_size, state_shape, actions_size):
        self.buffer_size    = buffer_size
        self.state_shape    = state_shape
        self.actions_size   = actions_size

        self.clear()

    def add(self, state, logits, value, action, reward, done):
        
        if self.ptr < self.buffer_size:

            self.states_b[self.ptr]    = state
            self.logits_b[self.ptr]    = logits
            self.values_b[self.ptr]    = value
            self.actions_b[self.ptr]   = action
            self.rewards_b[self.ptr]   = reward
            self.dones_b[self.ptr]     = done
        
            self.ptr = self.ptr + 1 

    def is_full(self):
        if self.ptr >= self.buffer_size:
            return True

        return False 
 
    def clear(self):
        self.states_b           = numpy.zeros((self.buffer_size, ) + self.state_shape, dtype=float)
        self.logits_b           = numpy.zeros((self.buffer_size, self.actions_size), dtype=float)
        
        self.values_b           = numpy.zeros((self.buffer_size, ), dtype=float)

        self.actions_b          = numpy.zeros((self.buffer_size, ), dtype=int)
        self.rewards_b          = numpy.zeros((self.buffer_size, ), dtype=float)
        self.dones_b            = numpy.zeros((self.buffer_size, ), dtype=float)
       
        self.returns_b          = numpy.zeros((self.buffer_size, ), dtype=float)
        self.advantages_b       = numpy.zeros((self.buffer_size, ), dtype=float)

        self.ptr = 0 


    def compute_gae_returns(self, gamma, lam = 0.95):
           
        gae  = 0.0
        for n in reversed(range(len(self.rewards_b)-1)):
            if self.dones_b[n] > 0:
                gamma_ = 0.0
            else:
                gamma_ = gamma

            delta = self.rewards_b[n] + gamma_*self.values_b[n+1] - self.values_b[n]

            gae = delta + lam*gamma_*gae

            self.returns_b[n] = gae + self.values_b[n]

        self.advantages_b = self.returns_b - self.values_b
        self.advantages_b = (self.advantages_b - numpy.mean(self.advantages_b))/(numpy.std(self.advantages_b) + 1e-10)


    def sample_batch(self, batch_size, device):
        indices         = torch.from_numpy(numpy.random.randint(0, self.buffer_size,    size=batch_size))

        states      = torch.from_numpy(numpy.take(self.states_b,     indices,  axis=0)).to(device).float()
        logits      = torch.from_numpy(numpy.take(self.logits_b,     indices,  axis=0)).to(device).float()
        values      = torch.from_numpy(numpy.take(self.values_b,     indices,  axis=0)).to(device).float()
        actions     = torch.from_numpy(numpy.take(self.actions_b,    indices,  axis=0)).to(device)
        rewards     = torch.from_numpy(numpy.take(self.rewards_b,    indices,  axis=0)).to(device).float()
        dones       = torch.from_numpy(numpy.take(self.dones_b,      indices,  axis=0)).to(device).float()
        returns     = torch.from_numpy(numpy.take(self.returns_b,    indices,  axis=0)).to(device).float()
        advantages  = torch.from_numpy(numpy.take(self.advantages_b, indices,  axis=0)).to(device).float()

        return states, logits, values, actions, rewards, dones, returns, advantages