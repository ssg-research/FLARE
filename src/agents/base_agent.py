############################################################################
# Modified from https://github.com/williamd4112/RL-Adventure               #
############################################################################

import os
import glob
import re

class base_agent:

    def __init__(self, name, agent_mode, env, device, args):
        #if agent_mode == 'dqn' and args.use_dueling:
        #    model_name = 'ddqn'
        #else:
        self.env = env
        self.args = args 
        self.device = device
        model_name = agent_mode

        if args.game_mode == "train":
            self.model_path = "./output/" + args.env_name + "/" + model_name + "/" + args.game_mode + "/"
            if not os.path.isdir(self.model_path):
                os.makedirs(self.model_path)

        self.model_path = "./output/" + args.env_name + "/" + model_name + "/" + args.game_mode + "/" + name  
        self.defense_path = "./output/" + args.env_name + "/" + model_name + "/" + "fingerprint" + "/"
        if args.game_mode == "advtrain":
            self.model_path = "./output/" + args.env_name + "/" + model_name + "/" + "fingerprint" + "/" + name  
            self.defense_path = "./output/" + args.env_name + "/" + model_name + "/" + "fingerprint" + "/"            
        
        self.model_filename = os.path.basename(self.model_path)
        self.model_id = os.path.splitext(self.model_filename)[0]
        self.model_base_path = os.path.dirname(self.model_path)

    def save_run(self, score, step, run):
        #self.logger.add_score(score)
        #self.logger.add_step(step)
        #self.logger.add_run(run)
        pass

    def select_action(self, obs, explore_eps=0.5, rnn_hxs=None, masks=None, deterministic=False):
        pass

    def remember(self,obs, action, reward, next_obs, done):
        pass

    def update_agent(self, total_step, rollouts=None, advmask=None):
        pass

    def update_model_path(self, f):
        self.model_path = f
        self.model_filename = os.path.basename(self.model_path)
        self.model_id = os.path.splitext(self.model_filename)[0]
        self.model_base_path = os.path.dirname(self.model_path)

    def get_model_path(self):
        return self.model_path
