import numpy
import torch

from torch.distributions import Categorical

from .PolicyBuffer import *

class AgentPPO():
    def __init__(self, envs, Model, Config):
        self.envs = envs

        config = Config.Config()

        self.gamma              = config.gamma
        self.entropy_beta       = config.entropy_beta
        self.eps_clip           = config.eps_clip

        self.steps              = config.steps
        self.batch_size         = config.batch_size        
        
        self.training_epochs    = config.training_epochs
        self.actors             = config.actors

        self.state_shape    = self.envs[0].observation_space.shape
        self.actions_count  = self.envs[0].action_space.n

        self.model          = Model.Model(self.state_shape, self.actions_count)
        self.optimizer      = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
 
        self.policy_buffer = PolicyBuffer(self.steps, self.state_shape, self.actions_count, self.actors, self.model.device)

        self.states = []
        for e in range(self.actors):
            self.states.append(self.envs[e].reset())

        self.enable_training()
        self.iterations = 0


    def enable_training(self):
        self.enabled_training = True

    def disable_training(self):
        self.enabled_training = False

    def main(self):
        rewards = []
        dones   = []

        states_t            = torch.tensor(self.states, dtype=torch.float).detach().to(self.model.device)
 
        logits_t, values_t  = self.model.forward(states_t)

        for e in range(self.actors):
            action = self._sample_action(logits_t[e])
                
            state, reward, done, _ = self.envs[e].step(action)
            
            if self.enabled_training:
                self.policy_buffer.add(e, states_t[e], logits_t[e], values_t[e], action, reward, done)

                if self.policy_buffer.is_full():
                    self.train()
                    
            if done:
                state = self.envs[e].reset()

            self.states[e] = state

            rewards.append(reward)
            dones.append(dones)

        self.iterations+= 1
        return rewards[0], dones[0]
    
    def save(self, save_path):
        self.model.save(save_path + "trained/")

    def load(self, save_path):
        self.model.load(save_path + "trained/")
    
    def _sample_action(self, logits):
        action_probs_t        = torch.nn.functional.softmax(logits.squeeze(0), dim = 0)
        action_distribution_t = torch.distributions.Categorical(action_probs_t)
        action_t              = action_distribution_t.sample()

        return action_t.item() 
    
    def train(self): 
        self.policy_buffer.compute_gae_returns(self.gamma)

        for e in range(self.training_epochs):
            states, logits, values, actions, rewards, dones, returns, advantages = self.policy_buffer.sample_batch(self.batch_size)

            loss = self._compute_loss(states, logits, actions, returns, advantages)

            self.optimizer.zero_grad()        
            loss.backward()
            for param in self.model.parameters():
                param.grad.data.clamp_(-0.5, 0.5)
            self.optimizer.step() 

        self.policy_buffer.clear()   

    
    def _compute_loss(self, states, logits, actions, returns, advantages):
        probs_old     = torch.nn.functional.softmax(logits, dim = 1).detach()
        log_probs_old = torch.nn.functional.log_softmax(logits, dim = 1).detach()

        logits_new, values_new   = self.model.forward(states)

        probs     = torch.nn.functional.softmax(logits_new, dim = 1)
        log_probs = torch.nn.functional.log_softmax(logits_new, dim = 1)

        '''
        compute critic loss, as MSE
        L = (T - V(s))^2
        '''
        loss_value = (returns.detach() - values_new)**2
        loss_value = loss_value.mean()

        ''' 
        compute actor loss, surrogate loss
        '''
        log_probs_      = log_probs[range(len(log_probs)), actions]
        log_probs_old_  = log_probs_old[range(len(log_probs_old)), actions]
                        
        ratio       = torch.exp(log_probs_ - log_probs_old_)
        p1          = ratio*advantages.detach()
        p2          = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip)*advantages.detach()
        loss_policy = -torch.min(p1, p2)  
        loss_policy = loss_policy.mean()
    
        '''
        compute entropy loss, to avoid greedy strategy
        L = beta*H(pi(s)) = beta*pi(s)*log(pi(s))
        '''
        loss_entropy = (probs*log_probs).sum(dim = 1)
        loss_entropy = self.entropy_beta*loss_entropy.mean()

        loss = loss_value + loss_policy + loss_entropy

        return loss
