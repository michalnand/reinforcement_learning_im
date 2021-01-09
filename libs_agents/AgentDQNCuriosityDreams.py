import numpy
import torch
from .ExperienceBuffer import *

import cv2


class AgentDQNCuriosityDreams():
    def __init__(self, env, ModelDQN, ModelAutoencoder, Config):
        self.env = env
 
        config = Config.Config()

        self.batch_size         = config.batch_size
        self.exploration        = config.exploration
        self.gamma              = config.gamma

        self.target_update      = config.target_update
        self.update_frequency   = config.update_frequency    


        self.beta               = config.beta
               
        self.state_shape    = self.env.observation_space.shape
        self.actions_count  = self.env.action_space.n

        self.experience_replay = ExperienceBuffer(config.experience_replay_size, self.state_shape, self.actions_count)

        self.model_dqn          = ModelDQN.Model(self.state_shape, self.actions_count)
        self.model_dqn_target   = ModelDQN.Model(self.state_shape, self.actions_count)
        self.optimizer_dqn      = torch.optim.Adam(self.model_dqn.parameters(), lr= config.learning_rate_dqn)

        for target_param, param in zip(self.model_dqn_target.parameters(), self.model_dqn.parameters()):
            target_param.data.copy_(param.data)


        self.model_autoencoder       = ModelAutoencoder.Model(self.state_shape)
        self.optimizer_autoencoder   = torch.optim.Adam(self.model_autoencoder.parameters(), lr= config.learning_rate_autoencoder)


        self.state    = env.reset()

        self.iterations     = 0

        self.enable_training()

        self.episodic_memory_size   = config.episodic_memory_size 
        self._init_episodic_memory(self.state)

        self.loss_autoencoder        = 0.0
        self.internal_motivation      = 0.0

    def enable_training(self):
        self.enabled_training = True

    def disable_training(self):
        self.enabled_training = False
    
    def main(self, show_activity = False):
        if self.enabled_training:
            self.exploration.process()
            epsilon = self.exploration.get()
        else:
            epsilon = self.exploration.get_testing()
                     
        state_t     = torch.from_numpy(self.state).to(self.model_dqn.device).unsqueeze(0).float()
        q_values_t  = self.model_dqn(state_t)
        q_values_np = q_values_t.squeeze(0).detach().to("cpu").numpy()

        action      = self._sample_action(q_values_np, epsilon)

        state_new, self.reward, done, self.info = self.env.step(action)

        if self.enabled_training:
            motivation = self._internal_motivation(self.state)
            self.experience_replay.add(self.state, action, self.reward, done, motivation)

        if self.enabled_training and (self.iterations > self.experience_replay.size):
            if self.iterations%self.update_frequency == 0:
                self.train_model()

            if self.iterations%self.target_update == 0:
                self.model_dqn_target.load_state_dict(self.model_dqn.state_dict())

        if done:
            self.state = self.env.reset()
            self._init_episodic_memory(self.state)
        else:
            self.state = state_new.copy()

        if show_activity:
            self._show_activity(self.state)

        self.iterations+= 1

        return self.reward, done
        
    def train_model(self):
        state_t, state_next_t, action_t, reward_t, done_t, motivation_t = self.experience_replay.sample(self.batch_size, self.model_dqn.device)

        #q values, state now, state next
        q_predicted      = self.model_dqn.forward(state_t)
        q_predicted_next = self.model_dqn_target.forward(state_next_t)

        #compute target, n-step Q-learning
        q_target         = q_predicted.clone()
        for j in range(self.batch_size): 
            action_idx              = action_t[j]
            q_target[j][action_idx] = reward_t[j] + motivation_t[j] + self.gamma*torch.max(q_predicted_next[j])*(1- done_t[j])
 
        #train DQN model
        loss_dqn  = ((q_target.detach() - q_predicted)**2)
        loss_dqn  = loss_dqn.mean() 

        self.optimizer_dqn.zero_grad()
        loss_dqn.backward()
        for param in self.model_dqn.parameters():
            param.grad.data.clamp_(-10.0, 10.0)
        self.optimizer_dqn.step()

        #train autoencoder prediction model
        states_predicted_t, _ = self.model_autoencoder(state_t)
 
        loss_autoencoder   = (state_t.detach() - states_predicted_t)**2
        loss_autoencoder   = loss_autoencoder.mean()

        self.optimizer_autoencoder.zero_grad()
        loss_autoencoder.backward()
        self.optimizer_autoencoder.step()

        k = 0.02
        self.loss_autoencoder       = (1.0 - k)*self.loss_autoencoder      + k*loss_autoencoder.detach().to("cpu").numpy()
        self.internal_motivation    = (1.0 - k)*self.internal_motivation    + k*motivation_t.mean().detach().to("cpu").numpy()

        print(">>> ", self.loss_autoencoder, self.internal_motivation)
    


    def save(self, save_path):
        self.model_dqn.save(save_path + "trained/")
        self.model_autoencoder.save(save_path + "trained/")

    def load(self, save_path):
        self.model_dqn.load(save_path + "trained/")
        self.model_autoencoder.load(save_path + "trained/")

    def get_log(self):
        result = "" 
        result+= str(round(self.loss_autoencoder, 7)) + " "
        result+= str(round(self.internal_motivation, 7)) + " "
        return result
    
    def _sample_action(self, q_values, epsilon):
        if numpy.random.rand() < epsilon:
            action_idx = numpy.random.randint(self.actions_count)
        else:
            action_idx = numpy.argmax(q_values)

        return action_idx

    def _show_activity(self, state, alpha = 0.6):
        activity_map    = self.model_dqn.get_activity_map(state)
        activity_map    = numpy.stack((activity_map,)*3, axis=-1)*[0, 0, 1]

        state_map    = numpy.stack((state[0],)*3, axis=-1)
        image        = alpha*state_map + (1.0 - alpha)*activity_map

        image        = (image - image.min())/(image.max() - image.min())

        image = cv2.resize(image, (400, 400), interpolation = cv2.INTER_AREA)
        cv2.imshow('state activity', image)
        cv2.waitKey(1)

    def _internal_motivation(self, state):
        state_t                     = torch.from_numpy(state).to(self.model_dqn.device).unsqueeze(0).float()

        #compute features for current state
        features_t                  = self.model_autoencoder.eval_features(state_t)
        features_t                  = features_t.view(features_t.size(0), -1).squeeze(0)

        features_np                 = features_t.detach().to("cpu").numpy()

        #find minimum distance from episodic memory
        distances_feature   = ((features_np - self.episodic_memory)**2).mean(axis=1)
        distance        = numpy.min(distances_feature)
        #distance        = distances_feature.mean()

        #compute internal motivation, higher distance, higher motivation
        motivation = numpy.tanh(self.beta*distance)

        #put current state into episodic memory, on random place
        idx = numpy.random.randint(self.episodic_memory_size)
        self.episodic_memory[idx] = features_np.copy()

        return motivation

    '''
    #TODO, KL divergence ditrubution comparing method
    def _internal_motivation_b(self, state):
        state_t                     = torch.from_numpy(state).to(self.model_dqn.device).unsqueeze(0).float()

        #compute features for current state
        features_t                  = self.model_autoencoder.eval_features(state_t)
        features_t                  = features_t.view(features_t.size(0), -1),squeeze(0)

        features_np                 = features_t.detach().to("cpu").numpy()

        #find current feature distance mean and std from episodic memory
        distances_feature       = ((features_np - self.episodic_memory)**2).mean(axis=1)
        distances_feature_mean  = distances_feature.mean()
        distances_feature_std   = distances_feature.std()

        #find old features distances mean and std from episodic memory
        distances_features_buffer = numpy.zeros((self.episodic_memory_size, self.episodic_memory_size))
        for i in range(self.episodic_memory_size):
            d = ((self.episodic_memory - self.episodic_memory[i])**2).mean(axis=1)
        distances_features_buffer[i] = d

        distances_features_mean = distances_features_buffer.mean()
        distances_features_std  = distances_features_buffer.std()

        #compute internal motivation, KL divergence
        motivation = self.beta*self._kl_divergence(distances_feature_mean, distances_feature_std, distances_features_mean, distances_features_std)

        #put current state into episodic memory, on random place
        idx = numpy.random.randint(self.episodic_memory_size)
        self.episodic_memory[idx] = features_np

        return motivation
    '''

   
    def _init_episodic_memory(self, state):   
        state_t                     = torch.from_numpy(state).to(self.model_dqn.device).unsqueeze(0).float()
        
        features_t                  = self.model_autoencoder.eval_features(state_t)
        features_t                  = features_t.view(features_t.size(0), -1)

        features_np                 = features_t.detach().to("cpu").numpy()

        self.episodic_memory        = numpy.tile(features_np, (self.episodic_memory_size, 1))
    
    def _kl_divergence(self, mu0, std0, mu1, std1):
        result = numpy.log(std1/(std0 + 0.000001))
        result+= ((std0**2) + (mu0 - mu1)**2)/(2*(std1**2))
        result+= -0.5

        return result