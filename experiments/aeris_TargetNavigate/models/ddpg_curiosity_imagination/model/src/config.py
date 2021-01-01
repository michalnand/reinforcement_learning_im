import libs_common.decay

class Config():

    def __init__(self):        
        self.gamma                  = 0.99
        self.critic_learning_rate   = 0.0002
        self.actor_learning_rate    = 0.0001
        self.forward_learning_rate  = 0.0002
        self.tau                    = 0.001

        self.beta_curiosity         = 10.0
        self.beta_imagination       = 10.0

        self.rollouts               = 16

        self.batch_size          = 64
        self.update_frequency    = 4

        self.exploration   = libs_common.decay.Const(0.05, 0.05)

        self.experience_replay_size = 200000

