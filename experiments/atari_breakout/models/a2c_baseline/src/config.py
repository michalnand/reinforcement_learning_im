class Config(): 
    def __init__(self):
        self.gamma                  = 0.99
        self.entropy_beta           = 0.001

        self.steps                  = 128
        self.batch_size             = 32
        
        self.actors                 = 8
        
        self.learning_rate          = 0.00025
