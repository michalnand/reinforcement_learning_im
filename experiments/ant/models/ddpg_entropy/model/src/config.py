import libs_common.decay

class Config():

    def __init__(self):        
        self.gamma                      = 0.99
        self.critic_learning_rate       = 0.0002
        self.actor_learning_rate        = 0.0001
        self.forward_learning_rate      = 0.0004
        self.autoencoder_learning_rate  = 0.0002

        self.tau                    = 0.001 
        self.beta1                  = 10.0
        self.beta2                  = 0.01

        self.batch_size          = 64
        self.update_frequency    = 4

        self.episodic_memory_size   = 256

        self.exploration   = libs_common.decay.Linear(1000000, 0.3, 0.05, 0.05)

        self.experience_replay_size = 200000
