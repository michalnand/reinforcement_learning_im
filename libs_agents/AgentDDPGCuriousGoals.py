import numpy
import torch
from .ExperienceBufferContinuousGoals import *


class AgentDDPGCuriousGoals():
    def __init__(self, env, ModelCritic, ModelActor, ModelForward, Config):
        self.env = env

        config = Config.Config()

        self.batch_size     = config.batch_size
        self.gamma          = config.gamma
        self.update_frequency = config.update_frequency
        self.tau                =  config.tau

        self.exploration    = config.exploration
    
        self.state_shape    = self.env.observation_space.shape
        self.actions_count  = self.env.action_space.shape[0]

        self.experience_replay = ExperienceBufferContinuousGoals(config.experience_replay_size, self.state_shape, self.actions_count)

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


        self.model_forward      = ModelForward.Model(self.state_shape, self.actions_count)
        self.optimizer_forward  = torch.optim.Adam(self.model_forward.parameters(), lr=config.forward_learning_rate)

        self.state              = env.reset()
        self.iterations         = 0

        self.loss_forward             = 0.0
        self.internal_motivation      = 0.0

        self.goal           = self.experience_replay.get_goal_by_motivation()

        self.enable_training()

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
        goal_t      = torch.from_numpy(self.goal).to(self.model_actor.device).unsqueeze(0).float()
        action_t, action = self._sample_action(state_t, goal_t, epsilon)
 
        action = action.squeeze()

        state_next, self.reward, done, self.info = self.env.step(action)

        if self.enabled_training: 

            state_next_predicted_t = self.model_forward(state_t, action_t)
            state_next_predicted_np= state_next_predicted_t.squeeze(0).detach().to("cpu").numpy()

            curiosity   = ((state_next - state_next_predicted_np)**2).mean()

            self.experience_replay.add(self.state, action, self.reward, done, curiosity)

        if self.enabled_training and self.iterations > 0.1*self.experience_replay.size:
            if self.iterations%self.update_frequency == 0:
                self.train_model()

        if done:
            self.state = self.env.reset()
            self.goal  = self.experience_replay.get_goal_by_motivation()
        else:
            self.state = state_next.copy()

        self.iterations+= 1

        return self.reward, done
        
        
    def train_model(self):
        state_t, state_next_t, action_t, reward_t, done_t, goals_t, motivation_t = self.experience_replay.sample(self.batch_size, self.model_critic.device)
        
        reward_t = reward_t.unsqueeze(-1)
        done_t   = (1.0 - done_t).unsqueeze(-1)

        action_next_t   = self.model_actor_target.forward(state_next_t, goals_t).detach()
        value_next_t    = self.model_critic_target.forward(state_next_t, goals_t, action_next_t).detach()

        motivation_t = self.beta*motivation_t

        #critic loss
        value_target    = reward_t + motivation_t + self.gamma*done_t*value_next_t
        value_predicted = self.model_critic.forward(state_t, goals_t, action_t)

        critic_loss     = ((value_target - value_predicted)**2)
        critic_loss     = critic_loss.mean()
     
        #update critic
        self.optimizer_critic.zero_grad()
        critic_loss.backward() 
        self.optimizer_critic.step()

        #actor loss
        actor_loss      = -self.model_critic.forward(state_t, goals_t, self.model_actor.forward(state_t, goals_t))
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



        #train forward model, MSE loss
        state_next_predicted_t  = self.model_forward(state_t, action_t)
        loss_forward = (state_next_t - state_next_predicted_t)**2
        loss_forward = loss_forward.mean() 

        self.optimizer_forward.zero_grad()
        loss_forward.backward()
        self.optimizer_forward.step()

        internal_motivation = motivation_t.mean().detach().to("cpu").numpy()

        k = 0.02
        self.loss_forward           = (1.0 - k)*self.loss_forward        + k*loss_forward.detach().to("cpu").numpy()
        self.internal_motivation    = (1.0 - k)*self.internal_motivation + k*internal_motivation



    def save(self, save_path):
        self.model_critic.save(save_path)
        self.model_actor.save(save_path)
        self.model_forward.save(save_path)

    def load(self, load_path):
        self.model_critic.load(load_path)
        self.model_actor.load(load_path)
        self.model_forward.load(load_path)

    
    def get_log(self):
        result = "" 
        result+= str(round(self.loss_forward, 7)) + " "
        result+= str(round(self.internal_motivation, 7)) + " "
        return result
    

    def _sample_action(self, state_t, goal_t, epsilon):
        action_t    = self.model_actor(state_t, goal_t)
        action_t    = action_t + epsilon*torch.randn(action_t.shape).to(self.model_actor.device)
        action_t    = action_t.clamp(-1.0, 1.0)

        action_np   = action_t.detach().to("cpu").numpy()

        return action_t, action_np
