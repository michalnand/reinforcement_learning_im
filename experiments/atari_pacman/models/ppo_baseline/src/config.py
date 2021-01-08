import libs_common.decay

class Config(): 

    def __init__(self):
        self.gamma                  = 0.99
        self.entropy_beta           = 0.01
        self.eps_clip               = 0.1
        self.batch_size             = 256
        self.episodes_to_train      = 16
        self.training_epochs        = 4
        self.learning_rate          = 0.00025
