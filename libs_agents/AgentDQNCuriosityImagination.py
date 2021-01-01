import numpy
import torch
from .ExperienceBuffer import *

class AgentDQNCuriosityImagination():
    def __init__(self, env, ModelDQN, ModelForward, ModelForwardTarget, Config):
        self.env    = env
 
        config      = Config.Config()

        self.batch_size         = config.batch_size
        self.exploration        = config.exploration
        self.gamma              = config.gamma
        self.beta_curiosity     = config.beta_curiosity
        self.beta_imagination   = config.beta_imagination
        self.rollouts           = config.rollouts
 
        self.target_update      = config.target_update
        self.update_frequency   = config.update_frequency        
               
        self.state_shape        = self.env.observation_space.shape
        self.actions_count      = self.env.action_space.n

        self.experience_replay  = ExperienceBuffer(config.experience_replay_size, self.state_shape, self.actions_count)

        self.model_dqn          = ModelDQN.Model(self.state_shape, self.actions_count)
        self.model_dqn_target   = ModelDQN.Model(self.state_shape, self.actions_count)
        self.optimizer_dqn      = torch.optim.Adam(self.model_dqn.parameters(), lr=config.learning_rate_dqn)

        for target_param, param in zip(self.model_dqn_target.parameters(), self.model_dqn.parameters()):
            target_param.data.copy_(param.data)

        self.model_forward          = ModelForward.Model(self.state_shape, self.actions_count)
        self.model_forward_target   = ModelForwardTarget.Model(self.state_shape, self.actions_count)
        self.optimizer_forward      = torch.optim.Adam(self.model_forward.parameters(), lr=config.learning_rate_forward)


        self.state              = env.reset()

        self.iterations         = 0

        self.loss_forward             = 0.0
        self.curiosity_motivation     = 0.0
        self.entropy_motivation       = 0.0
        
        self.enable_training()

        state_t        = torch.from_numpy(self.state).to(self.model_dqn.device).unsqueeze(0).float()
        self.curiosity_features_count = self.model_forward(state_t).shape[1]


    
    def enable_training(self):
        self.enabled_training = True

    def disable_training(self):
        self.enabled_training = False
    
    def main(self, show_activity = False):
        if self.enabled_training:
            self.exploration.process()
            self.epsilon = self.exploration.get()
        else:
            self.epsilon = self.exploration.get_testing()
             
        state_t        = torch.from_numpy(self.state).to(self.model_dqn.device).unsqueeze(0).float()

        action_idx_np, action_one_hot_t = self._sample_action(state_t, self.epsilon)

        self.action = action_idx_np[0]

        state_next, self.reward, done, self.info = self.env.step(self.action)

        if self.enabled_training:   
            self.experience_replay.add(self.state, self.action, self.reward, done)

        if self.enabled_training and (self.iterations > self.experience_replay.size):
            if self.iterations%self.update_frequency == 0:
                self.train_model()

            if self.iterations%self.target_update == 0:
                self.model_dqn_target.load_state_dict(self.model_dqn.state_dict())

        if done:
            self.state  = self.env.reset()
        else:
            self.state = state_next.copy()

        if show_activity:
            self._show_activity(self.state)

        self.iterations+= 1

        return self.reward, done

    def train_model(self):
        state_t, state_next_t, action_t, reward_t, done_t, _ = self.experience_replay.sample(self.batch_size, self.model_dqn.device)
 
        #curiosity internal motivation
        action_one_hot_t            = self._action_one_hot(action_t)
        curiosity_prediction_t      = self._curiosity(state_t, action_one_hot_t)
        curiosity_t                 = self.beta_curiosity*curiosity_prediction_t.detach()

        
        #train forward model, MSE loss
        loss_forward = curiosity_prediction_t.mean()
        self.optimizer_forward.zero_grad()
        loss_forward.backward()
        self.optimizer_forward.step()


        #entropy imagination motivation
        entropy_t = self.beta_imagination*self._imagination(state_t).detach()
       

        #q values, state now, state next
        q_predicted      = self.model_dqn.forward(state_t)
        q_predicted_next = self.model_dqn_target.forward(state_next_t)

        #compute target, n-step Q-learning
        q_target         = q_predicted.clone()
        for j in range(self.batch_size): 
            action_idx              = action_t[j] 
            q_target[j][action_idx] = reward_t[j] + curiosity_t[j] + entropy_t[j] + self.gamma*torch.max(q_predicted_next[j])*(1- done_t[j])
 
        #train DQN model
        loss_dqn  = (q_target.detach() - q_predicted)**2
        loss_dqn  = loss_dqn.mean() 

        self.optimizer_dqn.zero_grad()
        loss_dqn.backward()
        for param in self.model_dqn.parameters():
            param.grad.data.clamp_(-10.0, 10.0)
        self.optimizer_dqn.step()

        k = 0.02
        self.loss_forward           = (1.0 - k)*self.loss_forward           + k*loss_forward.detach().to("cpu").numpy()
        self.curiosity_motivation   = (1.0 - k)*self.curiosity_motivation   + k*curiosity_t.mean().detach().to("cpu").numpy()
        self.entropy_motivation     = (1.0 - k)*self.entropy_motivation     + k*entropy_t.mean().detach().to("cpu").numpy()

        #print(">>> ", self.loss_forward, self.curiosity_motivation, self.entropy_motivation)
 
    def save(self, save_path):
        self.model_dqn.save(save_path + "trained/")
        self.model_forward.save(save_path + "trained/")
        self.model_forward_target.save(save_path + "trained/")

    def load(self, save_path):
        self.model_dqn.load(save_path + "trained/")
        self.model_forward.load(save_path + "trained/")
        self.model_forward_target.save(save_path + "trained/")

    def get_log(self):
        result = "" 
        result+= str(round(self.loss_forward, 7)) + " "
        result+= str(round(self.curiosity_motivation, 7)) + " "
        result+= str(round(self.entropy_motivation, 7)) + " "
        return result

    def _action_one_hot(self, action_idx_t):
        batch_size = action_idx_t.shape[0]

        action_one_hot_t = torch.zeros((batch_size, self.actions_count))
        action_one_hot_t[range(batch_size), action_idx_t] = 1.0  
        action_one_hot_t = action_one_hot_t.to(self.model_dqn.device)

        return action_one_hot_t

    def _sample_action(self, state_t, epsilon):

        batch_size = state_t.shape[0]

        q_values_t          = self.model_dqn(state_t).to("cpu")

        #best actions indices
        q_max_indices_t     = torch.argmax(q_values_t, dim = 1)

        #random actions indices
        q_random_indices_t  = torch.randint(self.actions_count, (batch_size,))

        #create mask, which actions will be from q_random_indices_t and which from q_max_indices_t
        select_random_mask_t= torch.tensor((torch.rand(batch_size) < epsilon).clone(), dtype = int)

        #apply mask
        action_idx_t    = select_random_mask_t*q_random_indices_t + (1 - select_random_mask_t)*q_max_indices_t
        action_idx_t    = torch.tensor(action_idx_t, dtype=int)

        #create one hot encoding
        action_one_hot_t = torch.zeros((batch_size, self.actions_count))
        action_one_hot_t[range(batch_size), action_idx_t] = 1.0  
        action_one_hot_t = action_one_hot_t.to(self.model_dqn.device)

        #numpy result
        action_idx_np       = action_idx_t.detach().to("cpu").numpy().astype(dtype=int)

        return action_idx_np, action_one_hot_t

    def _curiosity(self, state_t, action_one_hot_t):
        state_next_predicted_t       = self.model_forward(state_t, action_one_hot_t)
        state_next_predicted_t_t     = self.model_forward_target(state_t, action_one_hot_t)

        curiosity_t    = (state_next_predicted_t_t.detach() - state_next_predicted_t)**2
        curiosity_t    = curiosity_t.mean(dim=1)

        return curiosity_t

    def _imagination(self, state_t):
        
        states_imagined_t = torch.zeros((self.rollouts, self.batch_size, self.curiosity_features_count)).to(state_t.device)
        for r in range(self.rollouts):
            _, action_one_hot_t  = self._sample_action(state_t, self.epsilon)
            states_imagined_t[r] = self.model_forward(state_t, action_one_hot_t)
        
        states_imagined_t = states_imagined_t.transpose(0, 1)
        entropy_t         = states_imagined_t.std(dim=1).mean(dim=1)
 
        return entropy_t
