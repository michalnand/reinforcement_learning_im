import numpy
import torch

from torch.distributions import Categorical

from .PolicyBuffer import *

class AgentPPO():
    def __init__(self, env, Model, Config):
        self.env = env

        config = Config.Config()
 
        self.gamma              = config.gamma
        self.entropy_beta       = config.entropy_beta
        self.eps_clip           = config.eps_clip

        self.ppo_steps              = config.ppo_steps
        self.batch_size             = config.batch_size
        self.training_epochs        = config.training_epochs


        self.state_shape    = self.env.observation_space.shape
        self.actions_count  = self.env.action_space.n

        self.model          = Model.Model(self.state_shape, self.actions_count)
        self.optimizer      = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
 
        self.policy_buffer = PolicyBuffer(self.ppo_steps, self.state_shape, self.actions_count)

        self.state = env.reset()

        self.enable_training()
        
        self.iterations = 0


    def enable_training(self):
        self.enabled_training = True

    def disable_training(self):
        self.enabled_training = False

    def main(self):
        state_t   = torch.tensor(self.state, dtype=torch.float32).detach().to(self.model.device).unsqueeze(0)
        
        logits_t, value_t   = self.model.forward(state_t)

        action = self._sample_action(logits_t)
            
        self.state, reward, done, _ = self.env.step(action)
        
        if self.enabled_training:
            state_np    = state_t.squeeze(0).detach().to("cpu").numpy()
            logits_np   = logits_t.squeeze(0).detach().to("cpu").numpy()
            value_np    = value_t.squeeze(0).detach().to("cpu").numpy()
            self.policy_buffer.add(state_np, logits_np, value_np, action, reward, done)

            if self.policy_buffer.is_full():
                self.train()
                  
        if done:
            self.state = self.env.reset()

            if hasattr(self.model, "reset"):
                self.model.reset()

        self.iterations+= 1

        return reward, done
    
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
            states, logits, values, actions, rewards, dones, returns, advantages = self.policy_buffer.sample_batch(self.batch_size, self.model.device)
            loss = self._compute_loss(states, logits, actions, returns, advantages)

            self.optimizer.zero_grad()        
            loss.backward()
            for param in self.model.parameters():
                param.grad.data.clamp_(-10.0, 10.0)
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
        loss_value = (returns - values_new)**2
        loss_value = loss_value.mean()

        ''' 
        compute actor loss, surrogate loss
        '''
        log_probs_      = log_probs[range(len(log_probs)), actions]
        log_probs_old_  = log_probs_old[range(len(log_probs_old)), actions]
                        
        ratio       = torch.exp(log_probs_ - log_probs_old_)
        p1          = ratio*advantages
        p2          = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip)*advantages
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
