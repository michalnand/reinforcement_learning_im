class Config(): 

    def __init__(self):
        self.gamma                  = 0.99
        self.learning_rate          = 0.0001 
        self.entropy_beta           = 0.01
       
        self.steps                  = 128
        self.batch_size             = 32 
        self.actors                 = 8
        
        