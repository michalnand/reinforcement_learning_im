import numpy
import torch

from torch.distributions import Categorical

from .PolicyBuffer import *

class AgentPPOCuriosity():
    def __init__(self, env, Model, ModelForward, ModelForwardTarget, Config):
        self.env = env

        config = Config.Config()
 
        self.gamma              = config.gamma
        self.entropy_beta       = config.entropy_beta
        self.eps_clip           = config.eps_clip

        self.ppo_steps              = config.ppo_steps
        self.batch_size             = config.batch_size
        self.training_epochs        = config.training_epochs

        self.beta               = config.beta


        self.state_shape    = self.env.observation_space.shape
        self.actions_count  = self.env.action_space.n

        self.model_ppo      = Model.Model(self.state_shape, self.actions_count)
        self.optimizer_ppo  = torch.optim.Adam(self.model_ppo.parameters(), lr=config.learning_rate_ppo)
 
        self.policy_buffer = PolicyBuffer(self.ppo_steps, self.state_shape, self.actions_count)


        self.model_forward          = ModelForward.Model(self.state_shape, self.actions_count)
        self.model_forward_target   = ModelForwardTarget.Model(self.state_shape, self.actions_count)
        self.optimizer_forward      = torch.optim.Adam(self.model_forward.parameters(), lr=config.learning_rate_forward)


        self.state = env.reset()

        self.enable_training()
        
        self.iterations = 0

        self.loss_forward             = 0.0
        self.internal_motivation      = 0.0


    def enable_training(self):
        self.enabled_training = True

    def disable_training(self):
        self.enabled_training = False

    def main(self):
        state_t   = torch.tensor(self.state, dtype=torch.float32).detach().to(self.model_ppo.device).unsqueeze(0)
        
        logits_t, value_t   = self.model_ppo.forward(state_t)

        action = self._sample_action(logits_t)
            
        self.state, reward, done, _ = self.env.step(action)
        
        if self.enabled_training:
            action_one_hot_t            = torch.zeros(1, self.actions_count).to(self.model_forward.device)
            action_one_hot_t[0][action] = 1.0

            curiosity_prediction_t      = self._curiosity(state_t, action_one_hot_t)
            curiosity_t                 = self.beta*curiosity_prediction_t
            curiosity                   = curiosity_t.detach().to("cpu").numpy()[0]
            
            state_np    = state_t.squeeze(0).detach().to("cpu").numpy()
            logits_np   = logits_t.squeeze(0).detach().to("cpu").numpy()
            value_np    = value_t.squeeze(0).detach().to("cpu").numpy()
            self.policy_buffer.add(state_np, logits_np, value_np, action, reward + curiosity, done)

            if self.policy_buffer.is_full():
                self.train()
                  
        if done:
            self.state = self.env.reset()

            if hasattr(self.model_ppo, "reset"):
                self.model_ppo.reset()

        self.iterations+= 1

        return reward, done
    
    def save(self, save_path):
        self.model_ppo.save(save_path + "trained/")
        self.model_forward.save(save_path + "trained/")
        self.model_forward_target.save(save_path + "trained/")

    def load(self, save_path):
        self.model_ppo.load(save_path + "trained/")
        self.model_forward.load(save_path + "trained/")
        self.model_forward_target.load(save_path + "trained/")

    def get_log(self):
        result = "" 
        result+= str(round(self.loss_forward, 7)) + " "
        result+= str(round(self.internal_motivation, 7)) + " "
        return result
    
    def _sample_action(self, logits):
        action_probs_t        = torch.nn.functional.softmax(logits.squeeze(0), dim = 0)
        action_distribution_t = torch.distributions.Categorical(action_probs_t)
        action_t              = action_distribution_t.sample()

        return action_t.item()
    
    def train(self): 

        self.policy_buffer.compute_gae_returns(self.gamma)

        for e in range(self.training_epochs):
            states, logits, values, actions, rewards, dones, returns, advantages = self.policy_buffer.sample_batch(self.batch_size, self.model_ppo.device)
             
            loss = self._compute_loss(states, logits, actions, returns, advantages)

            self.optimizer_ppo.zero_grad()        
            loss.backward()
            for param in self.model_ppo.parameters():
                param.grad.data.clamp_(-10.0, 10.0)
            self.optimizer_ppo.step() 


            #curiosity internal motivation
            action_one_hot_t            = self._action_one_hot(actions)
            curiosity_prediction_t      = self._curiosity(states, action_one_hot_t)
            curiosity_t                 = self.beta*curiosity_prediction_t

            #train forward model, MSE loss
            loss_forward = curiosity_prediction_t.mean()
            self.optimizer_forward.zero_grad()
            loss_forward.backward()
            self.optimizer_forward.step()

            k = 0.02
            self.loss_forward           = (1.0 - k)*self.loss_forward        + k*loss_forward.detach().to("cpu").numpy()
            self.internal_motivation    = (1.0 - k)*self.internal_motivation + k*curiosity_t.mean().detach().to("cpu").numpy()

           
        self.policy_buffer.clear()   

        print(">>> ", self.loss_forward, self.internal_motivation)
    

    
    def _compute_loss(self, states, logits, actions, returns, advantages):
        probs_old     = torch.nn.functional.softmax(logits, dim = 1).detach()
        log_probs_old = torch.nn.functional.log_softmax(logits, dim = 1).detach()

        logits_new, values_new   = self.model_ppo.forward(states)

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

    def _action_one_hot(self, action_idx_t):
        batch_size = action_idx_t.shape[0]

        action_one_hot_t = torch.zeros((batch_size, self.actions_count))
        action_one_hot_t[range(batch_size), action_idx_t] = 1.0  
        action_one_hot_t = action_one_hot_t.to(self.model_ppo.device)

        return action_one_hot_t

    def _curiosity(self, state_t, action_one_hot_t):
        state_next_predicted_t       = self.model_forward(state_t, action_one_hot_t)
        state_next_predicted_t_t     = self.model_forward_target(state_t, action_one_hot_t)

        curiosity_t    = (state_next_predicted_t_t.detach() - state_next_predicted_t)**2
        curiosity_t    = curiosity_t.mean(dim=1)

        return curiosity_t