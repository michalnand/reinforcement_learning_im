class Config(): 
    def __init__(self):
        self.gamma                  = 0.995
        self.entropy_beta           = 0.001
        self.eps_clip               = 0.1

        self.critic_loss_proportion = 0.5
        
        self.steps                  = 128
        self.batch_size             = 4
        
        self.training_epochs        = 4
        self.actors                 = 128
        
        self.learning_rate_ppo      = 0.0001
        self.learning_rate_forward  = 0.000025
        self.beta                   = 0.5
