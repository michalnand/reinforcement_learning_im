import threading
import multiprocessing
import time
import numpy
import gym

class MultiEnvSeq:
	def __init__(self, envs):
		self.envs	= envs
		
		self.obs 	= []
		self.reward = []
		self.done   = []
		self.info   = []

		for i in range(len(self.envs)):
			self.obs.append([])
			self.reward.append([])
			self.done.append([])
			self.info.append([])

	def close(self):
		pass

	def step(self, actions):
		for e in range(len(self.envs)):
			obs, reward, done, info = self.envs[e].step(actions[e])

			self.obs[e]		= obs
			self.reward[e]	= reward 
			self.done[e]	= done
			self.info[e]	= info
			
		return self.obs, self.reward, self.done, self.info

	def render(self):
		for i in range(len(self.envs)):
			self.envs[i].render()

	def __getitem__(self, idx):
		return self.envs[idx]



class MultiEnvParallel:
	def __init__(self, envs):
		self.envs 		= envs
		self.running 	= True

		threads_count_max 		= multiprocessing.cpu_count()
		threads_count_optimal   = max(len(self.envs)//16, 1)

		threads_count 			= min(threads_count_optimal, threads_count_max)


		print("MultiEnv ")
		print("trheads_count = ", threads_count)
		print("envs_count 	 = ", len(self.envs))
		print("\n\n")

		self.obs 	= []
		self.reward = []
		self.done   = []
		self.info   = []

		for i in range(len(self.envs)):
			self.obs.append([])
			self.reward.append([])
			self.done.append([])
			self.info.append([])

		envs_idx = []

		for i in range(threads_count):	
			envs_idx.append([])

		for i in range(len(self.envs)):
			envs_idx[i%threads_count].append(i)

		self.threads 	= []

		self.mutexes_in 	= []
		self.mutexes_out 	= []

		for i in range(threads_count):			
			th = threading.Thread(target=self._thread_main, args=(i, envs_idx[i]))
			self.threads.append(th) 
			self.mutexes_in.append(threading.Lock())
			self.mutexes_out.append(threading.Lock())

		for i in range(threads_count):
			self.mutexes_in[i].acquire()

		for i in range(len(self.mutexes_out)):
			self.mutexes_out[i].acquire()

		for i in range(threads_count):
			self.threads[i].start()

	def close(self):
		self.running = False

		for i in range(len(self.mutexes_in)):
			self.mutexes_in[i].release()

		for i in range(len(self.threads)):
			self.threads[i].join()

	def step(self, actions):
		self.actions = actions.copy()

		for i in range(len(self.mutexes_in)):
			self.mutexes_in[i].release()

		for i in range(len(self.mutexes_out)):
			self.mutexes_out[i].acquire()

		return self.obs, self.reward, self.done, self.info

	def render(self):
		for i in range(len(self.envs)):
			self.envs[i].render()

	def __getitem__(self, idx):
		return self.envs[idx]

	def _thread_main(self, id, envs_idx):
		while self.running == True:
			self.mutexes_in[id].acquire()

			for i in range(len(envs_idx)):
				env_idx = envs_idx[i]
				obs, reward, done, info = self.envs[env_idx].step(self.actions[env_idx])

				self.obs[env_idx] 		= obs
				self.reward[env_idx] 	= reward
				self.done[env_idx] 		= done
				self.info[env_idx] 		= info

			self.mutexes_out[id].release()



			
if __name__ == "__main__":

	envs = []

	n_envs = 128

	for i in range(n_envs):
		env = gym.make("MontezumaRevengeNoFrameskip-v4")
		env.reset()
		envs.append(env)

	#multi_envs = MultiEnvSeq(envs)
	multi_envs = MultiEnvParallel(envs)

	obs_shape = multi_envs[0].observation_space.shape
	n_actions = multi_envs[0].action_space.n


	print("INFO = ", obs_shape, n_actions)
	
	k = 0.02
	fps = 0
	steps = 0
	while True:
		
		actions = numpy.random.randint(n_actions, size=n_envs)

		time_start = time.time()
		obs, reward, done, info = multi_envs.step(actions)
		time_stop  = time.time()

		for j in range(len(done)):
			if done[j]:
				multi_envs[j].reset() 

		fps = (1.0 - k)*fps + k*1.0/(time_stop - time_start)

		if steps%100 == 0:	
			print("FPS = ", fps, fps*n_envs)

		steps+= 1
		
		#multi_envs.render()

	multi_envs.close()
