import RLAgents

class Config():

    def __init__(self):        
        self.gamma                          = 0.99
        self.critic_learning_rate           = 0.0002
        self.actor_learning_rate            = 0.0001
        self.forward_learning_rate          = 0.0002
        self.autoencoder_learning_rate      = 0.0002

        self.tau                            = 0.001 
        self.beta1                          = 1.0
        self.beta2                          = 10.0

        self.episodic_memory_size           = 64
        
        self.batch_size                     = 64
        self.update_frequency               = 4

        self.exploration   = RLAgents.DecayConst(0.1, 0.1)

        self.experience_replay_size = 200000
