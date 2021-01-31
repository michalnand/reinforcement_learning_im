import threading
import multiprocessing
import time
import numpy
import gym

class MultiEnvSeq:
	def __init__(self, env_name, wrapper, envs_count):

		dummy_env 	= gym.make(env_name)
		dummy_env 	= wrapper(dummy_env)

		self.observation_space 	= dummy_env.observation_space
		self.action_space 		= dummy_env.action_space

		self.envs	= []

		for i in range(envs_count):
			env 	= gym.make(env_name)
			env 	= wrapper(env)
			self.envs.append(env)

	def close(self):
		pass

	def reset(self, env_id):
		return self.envs[env_id].reset()

	def step(self, actions):
		obs 	= []
		reward 	= []
		done 	= []
		info 	= []

		for e in range(len(self.envs)):
			_obs, _reward, _done, _info = self.envs[e].step(actions[e])

			obs.append(_obs)
			reward.append(_reward)
			done.append(_done)
			info.append(_info)
			
		return obs, reward, done, info

	def render(self, env_id):
		for i in range(len(self.envs)):
			self.envs[i].render()
	
	def get(self, env_id):
		return self.envs[0]



def env_process_main(id, inq, outq, env_name, wrapper):

	print("env_process_main = ", id, env_name)

	env 	= gym.make(env_name)
	env  	= wrapper(env)
	
	obs 	= env.reset()
	reward 	= 0.0
	done 	= False
	info 	= None

	while True:
		val = inq.get()
		
		if val[0] == "end":
			break
		elif val[0] == "reset":
			obs 	= env.reset()
			reward 	= 0.0
			done 	= False
			info 	= None
			outq.put((obs, reward, done, info))

		elif val[0] == "step":
			action = val[1]
			obs, reward, done, info = env.step(action)
			outq.put((obs, reward, done, info))

		elif val[0] == "render":
			env.render()
			outq.put((obs, reward, done, info))

		elif val[0] == "get":
			outq.put(env)
	

class MultiEnvParallel:
	def __init__(self, env_name, wrapper, envs_count):

		dummy_env 	= gym.make(env_name)
		dummy_env	= wrapper(dummy_env)

		self.observation_space 	= dummy_env.observation_space
		self.action_space 		= dummy_env.action_space


		self.inq		= []
		self.outq 		= []
		self.workers 	= []


		for i in range(envs_count):
			inq	 =	multiprocessing.Queue()
			outq =	multiprocessing.Queue()

			worker = multiprocessing.Process(target=env_process_main, args=(i, inq, outq, env_name, wrapper))
			
			self.inq.append(inq)
			self.outq.append(outq)
			self.workers.append(worker) 

		for i in range(envs_count):
			self.workers[i].start()

	

	def close(self):
		for i in range(len(self.workers)):
			self.inq[i].put(["end"])
		
		for i in range(len(self.workers)):
			self.workers[i].join()

	def reset(self, env_id):
		self.inq[env_id].put(["reset"])

		obs, reward, done, info = self.outq[env_id].get()
		return obs 

	def render(self, env_id):
		self.inq[env_id].put(["render"])

		obs, reward, done, info = self.outq[env_id].get()

	def step(self, actions):
		for i in range(len(self.workers)):
			self.inq[i].put(["step", actions[i]])

		obs 	= []
		reward 	= []
		done   	= []
		info   	= []

		for i in range(len(self.workers)):
			_obs, _reward, _done, _info = self.outq[i].get()
			obs.append(_obs)
			reward.append(_reward)
			done.append(_done)
			info.append(_info)

		return obs, reward, done, info

	def get(self, env_id):
		self.inq[env_id].put(["get"])

		return self.outq[env_id].get()




			
if __name__ == "__main__":

	n_envs = 128

	#multi_envs = MultiEnvSeq("MontezumaRevengeNoFrameskip-v4", MontezumaWrapper, envs_count = n_envs)
	multi_envs = MultiEnvParallel("MontezumaRevengeNoFrameskip-v4", MontezumaWrapper, envs_count = n_envs)
	
	obs_shape = multi_envs.observation_space.shape
	n_actions = multi_envs.action_space.n

	print("INFO = ", obs_shape, n_actions)

	for i in range(n_envs):
		multi_envs.reset(i) 

	k 		= 0.02
	fps 	= 0
	steps 	= 0

	time_stop = 10
	time_start = 1
	while True:

		actions = numpy.random.randint(n_actions, size=n_envs)

		time_start = time.time()
		obs, rewards, dones, info = multi_envs.step(actions)
		time_stop  = time.time()

		for i in range(len(dones)):
			if dones[i]:
				multi_envs.reset(i) 


		fps = (1.0 - k)*fps + k*1.0/(time_stop - time_start)

		if steps%100 == 0:	
			print("FPS = ", fps, fps*n_envs)

		steps+= 1
		
		'''
		for i in range(n_envs):
			multi_envs.render(i)
		'''

	multi_envs.close()