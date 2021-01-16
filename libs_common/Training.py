import numpy
from .TrainingLog import *
import time

class TrainingEpisodes:
    def __init__(self, env, agent, episodes_count, episode_max_length, saving_path, logging_iterations = 10000, save_best_only = True):
        self.env = env
        self.agent = agent
        self.episodes_count = episodes_count
        self.episode_max_length = episode_max_length
        self.saving_path = saving_path
        self.logging_iterations = logging_iterations

        self.save_best_only = save_best_only

    def run(self):
        
        log = TrainingLog(self.saving_path + "result/result.log", self.logging_iterations, True)
        new_best = False
        iterations = 0
        for episode in range(self.episodes_count):

            self.env.reset()

            for i in range(self.episode_max_length):
                reward, done = self.agent.main()

            
                raw_episodes            = 0
                raw_score_per_episode   = 0
                raw_score_total         = 0

                if hasattr(self.env, "raw_episodes"):
                    raw_episodes = self.env.raw_episodes

                if hasattr(self.env, "raw_score_total"):
                    raw_score_total = self.env.raw_score_total

                if hasattr(self.env, "raw_score_per_episode"):
                    raw_score_per_episode = self.env.raw_score_per_episode

                log_agent = ""
                if hasattr(self.agent, "get_log"):
                    log_agent = self.agent.get_log() 

                log.add(reward, done, raw_episodes, raw_score_total, raw_score_per_episode, log_agent)

                if self.save_best_only == False:
                    new_best = True

                if log.is_best:
                    new_best = True
            
                iterations+= 1
                if iterations% self.logging_iterations == 0 and new_best == True:
                    new_best = False 
                    print("\n\n")
                    print("saving new best with score = ", log.episode_score_best)
                    self.agent.save(self.saving_path)
                    print("\n\n")

                if done:
                    break





class TrainingIterations:
    def __init__(self, env, agent, iterations_count, saving_path, saving_period_iterations = 10000, save_best_only = True):
        self.env = env
        self.agent = agent

        self.iterations_count = iterations_count
        
     
        self.saving_path = saving_path
        self.saving_period_iterations = saving_period_iterations

        self.save_best_only = save_best_only

    def run(self):
        log = TrainingLog(self.saving_path + "result/result.log", self.saving_period_iterations, True)
        new_best = False

        fps        = 0.0

        if isinstance(self.env, list):
            env = self.env[0]
        else:
            env = self.env

        for iteration in range(self.iterations_count):
            
            time_start      = time.time()
            reward, done    = self.agent.main()
            time_stop       = time.time()
            
            raw_episodes            = 0 
            raw_score_per_episode   = 0
            raw_score_total         = 0

            if hasattr(env, "raw_episodes"):
                raw_episodes = env.raw_episodes

            if hasattr(env, "raw_score_total"):
                raw_score_total = env.raw_score_total

            if hasattr(env, "raw_score_per_episode"):
                raw_score_per_episode = env.raw_score_per_episode

            log_agent = "" 
            if hasattr(self.agent, "get_log"):
                log_agent = self.agent.get_log() 

            log.add(reward, done, raw_episodes, raw_score_total, raw_score_per_episode, log_agent)

            
            if self.save_best_only == False:
                new_best = True

            if log.is_best:
                new_best = True
            
            if iteration%self.saving_period_iterations == 0 and new_best == True:
                new_best = False 
                print("\n\n")
                print("saving new best with score = ", log.episode_score_best)
                self.agent.save(self.saving_path)
                print("\n\n")

            k = 0.02
            fps = (1.0-k)*fps + k*1.0/(time_stop - time_start)
  
            if iteration%1000 == 0:

                time_remain = (self.iterations_count - iteration)/fps

                print("FPS = ",  round(fps, 3))
                print("ETA = ",  round(time_remain/3600, 2), " hours")
                print("\n\n")

            
        if new_best == True: 
            new_best = False 
            print("\n\n")
            print("saving new best with score = ", log.episode_score_best)
            self.agent.save(self.saving_path)
            print("\n\n")
