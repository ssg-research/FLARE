import glob
import os
import torch
import time

from pathlib import Path

from agents.dqn_agent import dqn_agent
from agents.a2c_agent import a2c_agent
from agents.ppo_agent import ppo_agent
import attack_utils.attacks as attacks


class attacker_agent:

    def __init__(self, env, agent, training_data_path, noise_path, device, args):
        self.collected_states = []
        self.collected_actions = []
        self.collected_action_distributions = []
        self.batch = 0
        self.total_observations = 0

        self.max_no_of_collected_states = 4000
        self.max_no_of_batches = 1
        self.fooling_rate = 0.0

        self.advmask = None
        self.training_data_path = training_data_path
        self.noise_path = noise_path
        self.device = device
        self.adversary = args.adversary
        self.eps = args.eps
        self.n_training_states = args.training_frames
        self.n_actions = env.action_space.n
        self.training_frames_type = args.training_frames_type

        self.idxs = []

        model_name = 'model.pt'
        agent_mode = args.attacker_agent_mode
        if agent_mode == 'dqn':
            self.proxy_agent = dqn_agent(model_name, agent_mode, env, device, args)
        elif agent_mode == 'a2c':
            self.proxy_agent = a2c_agent(model_name, agent_mode, env, device, args)
        elif agent_mode == 'ppo':
            self.proxy_agent = ppo_agent(model_name, agent_mode, env, device, args)
        self.proxy_agent.net.load_state_dict(agent.net.state_dict())

    def collect(self, obs, action, action_distribution): 
        if self.batch > self.max_no_of_batches:
            return
        # if adversary is random or plain fgsm, no need to collect any data
        if self.adversary == "random" or self.adversary == "none" or self.adversary == "deepfool":
            return
        self.collected_states.append(obs.squeeze(0))
        #if self.adversary == "uap":
        self.collected_action_distributions.append(action_distribution)

        # if the number of collected states are high, dump those into a file and increase batch by one. 
        if len(self.collected_states) >= self.max_no_of_collected_states:
            print("Number of training samples in {}-th batch is {}".format(self.batch, len(self.collected_states)))
            torch.save(torch.stack(self.collected_states).cpu(), self.training_data_path + "_X_batch_" + str(self.batch) + ".pt")
            self.batch += 1
            self.total_observations += len(self.collected_states)
            self.collected_states = []
            self.collected_actions = []
            self.collected_action_distributions = []

    def dump(self):
        # Check if there are collected game states, then dump those into a file
        if len(self.collected_states) > 0:
            self.total_observations += len(self.collected_states)
            torch.save(torch.stack(self.collected_states).cpu(), self.training_data_path + "_X_batch_" + str(self.batch) + ".pt")
            torch.save(torch.stack(self.collected_action_distributions).cpu(), self.training_data_path + "_c_batch_" + str(self.batch) + ".pt")
            torch.save(torch.tensor([self.total_observations]).cpu(), self.training_data_path + "_total_observations.pt")
            self.collected_states = []
            self.collected_actions = []
            self.collected_action_distributions = []

    def generate(self, obs, game_id, frame_idx_in_game, attacker_game_plays):

        # Generate mask when necessary
        if os.path.exists(self.noise_path) and frame_idx_in_game == 0 and game_id == attacker_game_plays:
            data = torch.load(self.noise_path, map_location=self.device)
            # load mask when the adversary is completely universal
            # e.g., uap_s, uap_f, obs_fgsm_wb 
            if data['iteration'] == -1 and self.adversary != 'obs_fgsm_wb_ingame':
                print("Loading an existing noise mask")
                self.advmask = data['advmask']
                return 

        if self.adversary == 'random' and frame_idx_in_game == 0 and game_id == attacker_game_plays:
            print("%s generates mask once at %d'th state in game %d" %(self.adversary, frame_idx_in_game, game_id))
            self.advmask = attacks.random_noise_attack(obs, self.eps, self.device)

        if (self.adversary == 'uaps' or self.adversary == 'uapo') and frame_idx_in_game == 0 and game_id == attacker_game_plays:
            print("%s generates mask once at %d'th state in game %d" %(self.adversary, frame_idx_in_game, game_id))
            print("size of training data before dump is %s" % len(self.collected_states))
            self.dump()
            print("size of training data after dump is %s" % len(self.collected_states))
            attacks.generate_uap(self.adversary, self.proxy_agent, self.noise_path, self.training_data_path, self.device, max_iter_uni=50,
                            xi =self.eps * 256, num_classes=self.n_actions, overshoot=0.2)
            self.advmask = torch.load(self.noise_path, map_location=self.device)['advmask']

    def attack(self, obs, frame_idx_in_game, agent):

        if self.adversary == 'uap' or self.adversary == 'osfwu' or self.adversary == 'cosfwu':
            start = time.time()
            obs_adv = torch.clamp(obs+self.advmask, 0.0, 255.0)
            end = time.time()
        elif self.adversary == 'random':
            start = time.time()
            obs_adv = torch.clamp(obs+self.advmask, 0.0, 255.0)
            end = time.time()
        else:
            start = time.time()
            obs_adv = obs.clone()
            end = time.time()
        return obs_adv, (end-start)


    def generate_osfwu_masks(self, sampling_strategy='none', slice_id=0):
        print(f"Length of collected states: {len(self.collected_states)}")
        self.advmask, self.fooling_rate, self.idxs = attacks.generate_osfw(self.proxy_agent, self.eps, self.collected_states,
                                                   self.device, self.n_training_states, sampling_strategy=sampling_strategy, slice_id=slice_id)

    def generate_conf_osfwu_masks(self, independent_agents, sampling_strategy='none', slice_id=0):
        print(f"Length of collected states: {len(self.collected_states)}")
        self.advmask, self.fooling_rate, self.idxs = attacks.generate_conferrable_osfw(self.proxy_agent, independent_agents, self.eps, self.collected_states,
                                                   self.device, self.n_training_states, sampling_strategy=sampling_strategy, slice_id=slice_id)

    def generate_uap_masks(self):
        print(f"Length of collected states: {len(self.collected_states)}")       
        self.advmask, self.fooling_rate, self.idxs = attacks.generate_uap(self.proxy_agent, self.eps, self.collected_states, 
                                                        self.collected_action_distributions, self.device)