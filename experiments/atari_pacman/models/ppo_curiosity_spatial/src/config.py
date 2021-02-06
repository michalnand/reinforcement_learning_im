class Config(): 
    def __init__(self):
        self.gamma                  = 0.99
        self.entropy_beta           = 0.001
        self.eps_clip               = 0.2

        self.steps                  = 128
        self.batch_size             = 32
        
        self.training_epochs        = 4
        self.actors                 = 1 #8
        
        self.learning_rate_ppo      = 0.00025
        self.learning_rate_forward  = 0.0001
        self.betas                  = [0.0025, 0.0025, 0.0025, 0.0025, 0.1]
        
