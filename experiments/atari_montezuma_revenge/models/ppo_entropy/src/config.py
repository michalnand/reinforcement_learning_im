class Config(): 
    def __init__(self):
        self.gamma                  = 0.99
        self.entropy_beta           = 0.001
        self.eps_clip               = 0.2

        self.steps                  = 128
        self.batch_size             = 4
        
        self.training_epochs        = 4
        self.actors                 = 128
        
        self.learning_rate_ppo          = 0.00025
        self.learning_rate_forward      = 0.000025
        self.learning_rate_autoencoder  = 0.00025
        
        self.beta1                      = 0.5
        self.beta2                      = 1.0

        self.episodic_memory_size       = 32
        