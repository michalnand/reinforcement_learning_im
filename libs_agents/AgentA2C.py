import numpy
import torch

from torch.distributions import Categorical

from .PolicyBuffer import *

class AgentA2C():
    def __init__(self, envs, Model, Config):
        self.envs       = envs
        self.envs_count = len(self.envs)

        config = Config.Config()
 
        self.gamma              = config.gamma
        self.entropy_beta       = config.entropy_beta
        self.batch_size         = config.batch_size
       
        self.state_shape    = self.envs[0].observation_space.shape
        self.actions_count  = self.envs[0].action_space.n

        self.model          = Model.Model(self.state_shape, self.actions_count)
        self.optimizer      = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
 
        self.policy_buffer = PolicyBuffer(self.batch_size, self.state_shape, self.actions_count, self.model.device, self.envs_count)

        self.states        = []

        for e in range(self.envs_count):
            self.states.append(envs[e].reset())

        self.enable_training()
        
        self.iterations = 0

    def enable_training(self):
        self.enabled_training = True

    def disable_training(self):
        self.enabled_training = False

    def main(self):
        rewards    = []
        dones      = []

        states_t            = torch.tensor(self.states, dtype=torch.float32).detach().to(self.model.device)
        logits, values      = self.model.forward(states_t)

        for e in range(self.envs_count):
            action = self._sample_action(logits[e])
                
            state, reward, done, _ = self.envs[e].step(action)

            if self.enabled_training:
                self.policy_buffer.add(e, states_t[e], logits[e], values[e], action, reward, done)  

            if done:
                state = self.envs[e].reset()
        
            self.states[e] = state     

            rewards.append(reward)
            dones.append(done)   
        
        self.iterations+= 1

        if self.policy_buffer.is_full():
            self._train()
            self.policy_buffer.clear()   
        
        return rewards[0], dones[0]
    
    def save(self, save_path):
        self.model.save(save_path)

    def load(self, save_path):
        self.model.load(save_path)
    
    def _sample_action(self, logits):
        action_probs_t        = torch.nn.functional.softmax(logits.squeeze(0), dim = 0)
        action_distribution_t = torch.distributions.Categorical(action_probs_t)
        action_t              = action_distribution_t.sample()

        return action_t.item()
  
    
    def _train(self):      
        self.policy_buffer.compute_gae_returns(self.gamma)

        loss = self._compute_loss()

        self.optimizer.zero_grad()        
        loss.backward()
        self.optimizer.step() 


    def _compute_loss(self):

        logits_b        = self.policy_buffer.logits_b.reshape(self.envs_count*self.batch_size, self.actions_count)
        returns_b       = self.policy_buffer.returns_b.reshape(self.envs_count*self.batch_size, ).detach()
        values_b        = self.policy_buffer.values_b.reshape(self.envs_count*self.batch_size, )
        advantages_b    = self.policy_buffer.advantages_b.reshape(self.envs_count*self.batch_size, ).detach()
        actions_b       = self.policy_buffer.actions_b.reshape(self.envs_count*self.batch_size, )

        probs     = torch.nn.functional.softmax(logits_b, dim = 1)
        log_probs = torch.nn.functional.log_softmax(logits_b, dim = 1)

        '''
        compute critic loss, as MSE
        L = (T - V(s))^2
        '''
        loss_value = (returns_b - values_b)**2
        loss_value = loss_value.mean()

        ''' 
        compute actor loss 
        L = log(pi(s, a))*(T - V(s)) = log(pi(s, a))*A 
        '''
        loss_policy = -log_probs[range(len(log_probs)), actions_b]*advantages_b
        loss_policy = loss_policy.mean()

        '''
        compute entropy loss, to avoid greedy strategy
        L = beta*H(pi(s)) = beta*pi(s)*log(pi(s))
        '''
        loss_entropy = (probs*log_probs).sum(dim = 1)
        loss_entropy = self.entropy_beta*loss_entropy.mean()

        loss = loss_value + loss_policy + loss_entropy

        return loss


    def _compute_loss(self, e):
        probs     = torch.nn.functional.softmax(self.policy_buffer.logits_b[e], dim = 1)
        log_probs = torch.nn.functional.log_softmax(self.policy_buffer.logits_b[e], dim = 1)

        '''
        compute critic loss, as MSE
        L = (T - V(s))^2
        '''
        loss_value = (self.policy_buffer.returns_b[e] - self.policy_buffer.values_b[e])**2
        loss_value = loss_value.mean()

        ''' 
        compute actor loss 
        L = log(pi(s, a))*(T - V(s)) = log(pi(s, a))*A 
        '''
        advantage   = self.policy_buffer.advantages_b[e]
        loss_policy = -log_probs[range(len(log_probs)), self.policy_buffer.actions_b[e]]*advantage
        loss_policy = loss_policy.mean()

        '''
        compute entropy loss, to avoid greedy strategy
        L = beta*H(pi(s)) = beta*pi(s)*log(pi(s))
        '''
        loss_entropy = (probs*log_probs).sum(dim = 1)
        loss_entropy = self.entropy_beta*loss_entropy.mean()

        loss = loss_value + loss_policy + loss_entropy

        return loss
