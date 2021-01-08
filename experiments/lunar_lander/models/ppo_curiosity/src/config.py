class Config(): 

    def __init__(self):
        self.gamma                  = 0.99
        self.entropy_beta           = 0.01
        self.eps_clip               = 0.2
        
        self.ppo_steps              = 2048
        self.batch_size             = 64
        self.training_epochs        = 10

        self.beta                   = 1.0

        self.learning_rate_ppo      = 0.00025
        self.learning_rate_forward  = 0.0004
