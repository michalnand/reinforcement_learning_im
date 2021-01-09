import numpy
import torch
from .ExperienceBufferContinuous import *


class AgentDDPGCuriosityEM():
    def __init__(self, env, ModelCritic, ModelActor, ModelReachability, Config):
        self.env = env

        config = Config.Config()

        self.batch_size         = config.batch_size
        self.gamma              = config.gamma
        self.update_frequency   = config.update_frequency
        self.tau                = config.tau
        self.beta               = config.beta

        self.exploration    = config.exploration
    
        self.state_shape    = self.env.observation_space.shape
        self.actions_count  = self.env.action_space.shape[0]

        self.experience_replay = ExperienceBufferContinuous(config.experience_replay_size, self.state_shape, self.actions_count)

        self.model_actor            = ModelActor.Model(self.state_shape, self.actions_count)
        self.model_actor_target     = ModelActor.Model(self.state_shape, self.actions_count)

        self.model_critic           = ModelCritic.Model(self.state_shape, self.actions_count)
        self.model_critic_target    = ModelCritic.Model(self.state_shape, self.actions_count)

        for target_param, param in zip(self.model_actor_target.parameters(), self.model_actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.model_critic_target.parameters(), self.model_critic.parameters()):
            target_param.data.copy_(param.data)

        self.optimizer_actor    = torch.optim.Adam(self.model_actor.parameters(), lr= config.actor_learning_rate)
        self.optimizer_critic   = torch.optim.Adam(self.model_critic.parameters(), lr= config.critic_learning_rate)

        self.model_reachability       = ModelReachability.Model(self.state_shape)
        self.optimizer_reachability   = torch.optim.Adam(self.model_reachability.parameters(), lr= config.learning_rate_reachability)

        self.state          = env.reset()

        self.iterations     = 0

        self.enable_training()

        self.episodic_memory_size   = config.episodic_memory_size 
        self._init_episodic_memory(self.state)

        self.loss_reachability        = 0.0
        self.internal_motivation      = 0.0

    def enable_training(self):
        self.enabled_training = True

    def disable_training(self):
        self.enabled_training = False
    
    def main(self):
        if self.enabled_training:
            self.exploration.process()
            epsilon = self.exploration.get()
        else:
            epsilon = self.exploration.get_testing()
       
        state_t     = torch.from_numpy(self.state).to(self.model_actor.device).unsqueeze(0).float()

        action_t, action = self._sample_action(state_t, epsilon)
 
        action = action.squeeze()

        state_new, reward, done, self.info = self.env.step(action)

        if self.enabled_training:
            _, _, motivation = self._reachability(state_t)
            self.experience_replay.add(self.state, action, reward, done, motivation)

        if self.enabled_training and self.iterations > 0.1*self.experience_replay.size:
            if self.iterations%self.update_frequency == 0:
                self.train_model()

        if done:
            self.state = self.env.reset()
        else:
            self.state = state_new.copy()

        self.iterations+= 1

        return reward, done
        
        
    def train_model(self):
        state_t, state_next_t, action_t, reward_t, done_t, motivation_t = self.experience_replay.sample(self.batch_size, self.model_critic.device)
        
        reward_t = reward_t.unsqueeze(-1)
        done_t   = (1.0 - done_t).unsqueeze(-1)

        action_next_t   = self.model_actor_target.forward(state_next_t).detach()
        value_next_t    = self.model_critic_target.forward(state_next_t, action_next_t).detach()

        #critic loss
        value_target    = reward_t + motivation_t + self.gamma*done_t*value_next_t
        value_predicted = self.model_critic.forward(state_t, action_t)

        critic_loss     = ((value_target - value_predicted)**2)
        critic_loss     = critic_loss.mean()
     
        #update critic
        self.optimizer_critic.zero_grad()
        critic_loss.backward() 
        self.optimizer_critic.step()

        #actor loss
        actor_loss      = -self.model_critic.forward(state_t, self.model_actor.forward(state_t))
        actor_loss      = actor_loss.mean()

        #update actor
        self.optimizer_actor.zero_grad()       
        actor_loss.backward()
        self.optimizer_actor.step()

        # update target networks 
        for target_param, param in zip(self.model_actor_target.parameters(), self.model_actor.parameters()):
            target_param.data.copy_((1.0 - self.tau)*target_param.data + self.tau*param.data)
       
        for target_param, param in zip(self.model_critic_target.parameters(), self.model_critic.parameters()):
            target_param.data.copy_((1.0 - self.tau)*target_param.data + self.tau*param.data)

        #train reachability prediction model
        states_a_t, states_b_t, reachability_t  = self.experience_replay.sample_reachable_pairs(self.batch_size, device=self.model_reachability.device)

        reachability_predicted_t = self.model_reachability(states_a_t, states_b_t)
 
        loss_reachability   = (reachability_t.detach() - reachability_predicted_t)**2
        loss_reachability   = loss_reachability.mean()

        self.optimizer_reachability.zero_grad()
        loss_reachability.backward()
        self.optimizer_reachability.step()

        k = 0.02
        self.loss_reachability      = (1.0 - k)*self.loss_reachability      + k*loss_reachability.detach().to("cpu").numpy()
        self.internal_motivation    = (1.0 - k)*self.internal_motivation    + k*motivation_t.mean().detach().to("cpu").numpy()

        #print(">>> ", loss_reachability, self.loss_reachability, self.internal_motivation)
    
    def save(self, save_path):
        self.model_critic.save(save_path)
        self.model_actor.save(save_path)

    def load(self, load_path):
        self.model_critic.load(load_path)
        self.model_actor.load(load_path)

    def get_log(self):
        result = "" 
        result+= str(round(self.loss_reachability, 7)) + " "
        result+= str(round(self.internal_motivation, 7)) + " "
        return result
    
    def _sample_action(self, state_t, epsilon):
        action_t    = self.model_actor(state_t)
        action_t    = action_t + epsilon*torch.randn(action_t.shape).to(self.model_actor.device)
        action_t    = action_t.clamp(-1.0, 1.0)

        action_np   = action_t.detach().to("cpu").numpy()

        return action_t, action_np

    def _reachability(self, state_t):
        #compute reachability_t, compare with episodic_memory_t
        if len(self.state_shape) == 1:
            state_tmp_t = state_t.repeat(self.episodic_memory_size, 1)
        elif len(self.state_shape) == 2:
            state_tmp_t = state_t.repeat(self.episodic_memory_size, 1, 1)
        else: 
            state_tmp_t = state_t.repeat(self.episodic_memory_size, 1, 1, 1)

        reachability_t = self.model_reachability(state_tmp_t, self.episodic_memory_t)
 
        reachability_np = reachability_t.detach().to("cpu").numpy()

        max_idx = numpy.argmax(reachability_np)

        motivation = self.beta*(0.5 - reachability_np[max_idx])
 
        #put current state into episodic memory, on random place
        idx = numpy.random.randint(self.episodic_memory_size)
        self.episodic_memory_t[idx] = state_t.clone()

        return reachability_np, max_idx, motivation
 
    def _init_episodic_memory(self, state):   
        state_t                     = torch.from_numpy(state).to(self.model_reachability.device).unsqueeze(0).float()
     
        if len(self.state_shape) == 1:
            self.episodic_memory_t      = state_t.repeat(self.episodic_memory_size, 1)
        elif len(self.state_shape) == 2:
            self.episodic_memory_t      = state_t.repeat(self.episodic_memory_size, 1, 1)
        else: 
            self.episodic_memory_t      = state_t.repeat(self.episodic_memory_size, 1, 1, 1)
