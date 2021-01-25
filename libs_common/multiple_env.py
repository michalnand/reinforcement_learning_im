import multiprocessing
import time
import numpy


class Worker:
	def __init__(self, env, id):
		self.env 		= env
		self.id 		= id
		self.running 	= True

	def process_main(self):
		while self.running == True:
			print("running ID ", self.id, self.running)
			time.sleep(0.1)

	def stop(self):
		self.running = False


class MultiEnv:
	def __init__(self, envs):
		self.envs 		= envs
		self.running 	= True

		self.workers 	= []
		self.process 	= []

		for i in range(len(envs)):
			w = Worker(self.envs[i], i)
			self.workers.append(w)

			p = multiprocessing.Process(target=w.process_main, args=( ))
			self.process.append(p) 

		for i in range(len(envs)):
			self.process[i].start()

	def close(self):
		for i in range(len(envs)):
			self.workers[i].stop()

		print("AAAAAAAAAAAA\n\n\n\n\n\n")

		for i in range(len(envs)):
			self.process[i].join()





if __name__ == "__main__":

	envs = [1, 2, 3, 4, 5]

	multi_envs = MultiEnv(envs)

	time.sleep(2)

	multi_envs.close()

	print("program done")