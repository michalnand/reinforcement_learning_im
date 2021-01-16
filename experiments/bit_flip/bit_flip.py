import gym
from gym import spaces
import numpy

class BitFlip:
    def __init__(self, size = 8):

        self.size = size

        self.action_space       = spaces.Discrete(self.size)
        self.observation_space  = spaces.Box(low=-1.0, high=1.0,  shape=(2, self.size), dtype=numpy.float32)

    def reset(self):
        self.target_state = numpy.random.randint(2, size=self.size)
        self.agent_state  = numpy.random.randint(2, size=self.size)

        self.steps = 0

        return self._update_observation()

    def step(self, action):
        self.agent_state[action] = 1 - self.agent_state[action]

      
        self.steps+= 1

        dif = ((self.agent_state - self.target_state)**2).max()

        if self.steps >= self.size:
            reward = -1.0
            done   = True
        elif dif < 0.1:
            reward = 1.0
            done   = True
        else:
            reward = 0.0
            done   = False


        return self._update_observation(), reward, done, None

    def render(self):
        print("steps                = ", self.steps)
        print(self.target_state)
        print(self.agent_state)
        print("\n\n")


    def _update_observation(self):
        result = numpy.zeros((2, self.size))

        result[0] = self.target_state.copy()
        result[1] = self.agent_state.copy()

        return result


if __name__ == "__main__":
    env     = ChartMove(size=256)
    state   = env.reset()

    score = 0.0
    while True:
        action = numpy.random.randint(3)
        state, reward, done, _ = env.step(action)

        score+= reward

        if done:
            state = env.reset()
            print("score = ", score)



    