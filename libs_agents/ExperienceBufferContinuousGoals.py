import numpy
import torch

class ExperienceBufferContinuousGoals():
    def __init__(self, size, state_shape, actions_count):

        self.size           = size       
        self.current_idx    = 0 
        self.initialized    = False

        self.state_shape        = state_shape
        self.actions_count      = actions_count

    def _initialize(self):
        if self.initialized == False:
            self.state_b        = numpy.zeros((self.size, ) + self.state_shape, dtype=numpy.float32)
            self.action_b       = numpy.zeros((self.size, self.actions_count), dtype=numpy.float32)
            self.reward_b       = numpy.zeros((self.size, ), dtype=numpy.float32)
            self.done_b         = numpy.zeros((self.size, ), dtype=numpy.float32)
            self.ir_b           = numpy.zeros((self.size, ), dtype=numpy.float32)

            self.initialized    = True

    def add(self, state, action, reward, done, ir = 0.0): 
        self._initialize()

        if done != 0: 
            done_ = 1.0
        else:
            done_ = 0.0

        self.state_b[self.current_idx]          = state.copy()
        self.action_b[self.current_idx]         = action.copy()
        self.reward_b[self.current_idx]         = reward
        self.done_b[self.current_idx]           = done_
        self.ir_b[self.current_idx]             = ir

        self.current_idx = (self.current_idx + 1)%self.size

    def sample(self, batch_size, device = "cpu"):
        indices         = numpy.random.randint(0, self.size, size=batch_size)
        indices_next    = (indices + 1)%self.size

        state_t         = torch.from_numpy(numpy.take(self.state_b,     indices, axis=0)).to(device)
        state_next_t    = torch.from_numpy(numpy.take(self.state_b,     indices_next, axis=0)).to(device)
        action_t        = torch.from_numpy(numpy.take(self.action_b,    indices, axis=0)).to(device)
        reward_t        = torch.from_numpy(numpy.take(self.reward_b,    indices, axis=0)).to(device)
        done_t          = torch.from_numpy(numpy.take(self.done_b,      indices, axis=0)).to(device)
        ir_t            = torch.from_numpy(numpy.take(self.ir_b,        indices, axis=0)).to(device)

        #create states sequence, starting from "now position", take sequence_length samples into past
        states_seq_t    = torch.zeros((self.sequence_length, batch_size) + self.state_shape).to(device)
        
        #values if intrinsics motivation
        ir_values       = numpy.zeros((self.sequence_length, batch_size))

        for j in range(self.sequence_length):
            indices_        = (indices - j)%self.size
            states_seq_t[j] = torch.from_numpy(numpy.take(self.state_b, indices_, axis=0)).to(device)
            
            ir_values[j]    = numpy.take(self.ir_b, indices_, axis=0)

        #transpose to shape : batch, sequence_length, state_shape
        states_seq_t = states_seq_t.transpose(0, 1)
        
        #transpose to shape : batch, sequence_length
        ir_values     = numpy.transpose(ir_values)

        #relative indices into absolute indices
        #take the state indices with highest ir
        goals_indices = (indices - numpy.argmax(ir_values, axis=1))%self.size

        #take goal state
        goal_t        = torch.from_numpy(numpy.take(self.state_b,     goals_indices, axis=0)).to(device)
        
        return state_t, state_next_t, states_seq_t, goal_t, action_t, reward_t, done_t, ir_t