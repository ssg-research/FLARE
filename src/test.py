# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import torch
import logging
import numpy as np
import torch.nn.functional as F

from agents.dqn_agent import dqn_agent
from agents.a2c_agent import a2c_agent
from agents.ppo_agent import ppo_agent

from agents.models import DQNnet
from rl_utils.utils import set_seeds
from rl_utils.atari_wrapper import make_vec_envs
import agents.action_conditional_video_prediction as acvp

def get_acvp_threshold(agent_mode, env_name):
    # return difference thresholds for the acvp module based on the agent mode 
    # thresholds are calculated to preserve the maximum performance of the ....
    # ... agent when there is no adversarial mask added to the state. 
    # calculated by np.sort(distances)[-1]*0.90, 0.05 rejection rate.
    if agent_mode == "dqn" and env_name == "Pong":
        return 0.050
    elif agent_mode == "dqn" and env_name == "MsPacman":
        return 0.045#0.050
    elif agent_mode == "a2c" and env_name == "Pong":
        return 0.015
    elif agent_mode == "a2c" and env_name == "MsPacman":
        return 0.040
    elif agent_mode == "ppo" and env_name == "Pong":
        return 0.020
    elif agent_mode == "ppo" and env_name == "MsPacman":
        return 0.050
    else:
        return 0.050

def acvp_action_correction(vf, action_dist, action):
    with torch.no_grad():
        action_dist = F.softmax(action_dist, dim=-1)
        predicted_next_dist = F.softmax(vf.predicted_next_dist, dim=-1)
        diff = abs(action_dist - predicted_next_dist).sum()/len(action_dist)
        action_match = int(action == vf.predicted_next_action)
    if diff > vf.diff_threshold:
        return vf.predicted_next_action, diff, action_match
    else:
        return action, diff, action_match

def acvp2_action_correction(vf, action_dist, action, rar=0.0):
    # Create a random number between 0 and 100
    random_number = np.random.randint(0, 100)
    with torch.no_grad():
        action_dist = F.softmax(action_dist, dim=-1)
        predicted_next_dist = F.softmax(vf.predicted_next_dist, dim=-1)
        diff = abs(action_dist - predicted_next_dist).sum()/len(action_dist)
        action_match = int(action == vf.predicted_next_action)
    if diff > vf.diff_threshold and random_number <= rar*100 and random_number!=0.0:
        sorted_actions = predicted_next_dist.sort(descending=True)[1]
        sorted_actions = sorted_actions[sorted_actions!=vf.predicted_next_action.item()]
        return sorted_actions[0], diff, action_match
    else:
        if diff > vf.diff_threshold:
            return vf.predicted_next_action, diff, action_match
        else:
            return action, diff, action_match

def acvp_next_state_prediction(vf, action_one_hot, action, next_state):
    with torch.no_grad():
        action_one_hot[action_one_hot!=0] = 0
        action_one_hot[action] = 1
        vf.predicted_next_state = next_state.clone()
        temp = acvp.post_process(vf.predict(acvp.pre_process(vf.predicted_next_state[:,:3,:,:], vf.mean_state), action_one_hot), vf.mean_state)
        vf.predicted_next_state[:,-1,:,:] = torch.round(temp.clone().detach())


def act(agent, agent_mode, masks, state, recurrent_hidden_states, deterministic=True, rar=0.0):
    if agent_mode == "imit":
        action_distribution = F.softmax(agent.forward(state))
        action = torch.argmax(action_distribution)
    if agent_mode == 'dqn':
        action, action_distribution = agent.select_action(state)
    elif agent_mode == 'a2c' or agent_mode == 'ppo':
        with torch.no_grad():
            _, action, _, action_distribution = agent.select_action(state, recurrent_hidden_states,
                                                                    masks, log_probs=True, deterministic=deterministic)
    # Create a random number between 0 and 100
    random_number = np.random.randint(0, 100)
    # If the random number is less than or equal to the percentage of randomness, return a random action
    if random_number <= rar*100 and random_number!=0.0:
        random_id = np.random.choice(action_distribution.size(0))
        action[0] = random_id
        temp1 = action_distribution[torch.argmax(action_distribution)].item()
        temp2 = action_distribution[random_id].item()
        action_distribution[torch.argmax(action_distribution)] = temp2
        action_distribution[random_id] = temp1

    return action, action_distribution

def construct_agent(model_type, env, device, args):

    if model_type == 'victim':
        agent_mode = args.victim_agent_mode
        if args.robust:
            model_name = "model_original_robust.pt"
        else:
            model_name = "model_original.pt"
    else:
        agent_mode = args.suspected_agent_mode
        model_name = args.suspected_agent_path.split('/')[-1]

    if agent_mode == 'dqn':
        agent = dqn_agent(model_name, agent_mode, env, device, args)
    elif agent_mode == 'a2c':
        agent = a2c_agent(model_name, agent_mode, env, device, args)
    elif agent_mode == 'ppo':
        agent = ppo_agent(model_name, agent_mode, env, device, args)

    if model_type == 'victim':
        if args.victim_agent_path != "":
            agent.model_path = args.victim_agent_path
        print(f"Load model from: {agent.model_path}")
        checkpoint = torch.load(agent.model_path, map_location=lambda storage, loc: storage)
    else:
        print(f"Load model from: {args.suspected_agent_path}")
        checkpoint = torch.load(args.suspected_agent_path, map_location=lambda storage, loc: storage)
        agent.model_path = args.suspected_agent_path

    if 'model_state_dict' in checkpoint:
        # load the state dictionary inside the checkpoint
        agent.net.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        # the checkpoint itself is the state dictionary
        agent.net.load_state_dict(torch.load(agent.model_path, map_location=lambda storage, loc: storage))
        
    agent.net.to(device)

    return agent

def test(args):

    logger = logging.getLogger('fingerprint_logger')
    logger.setLevel(logging.INFO)

    device = torch.device("cuda:0" if args.cuda else "cpu")
    # Environment, create
    env = make_vec_envs(args.env_name, args.seed + 1000, 1, None, 'output/env_logs', device=device,
                        allow_early_resets=args.allow_early_resets)
    # Environment, set seeds
    set_seeds(args, 1000)

    if args.suspected_agent_path == "":
        agent = construct_agent(model_type="victim", env=env, device=device, args=args)
        agent_mode = args.victim_agent_mode
        handler = logging.FileHandler(f"{args.env_name}fingerprint_{args.victim_agent_mode}_{args.adversary}.log")
        logger.addHandler(handler)
        logger.info("-----------------")
        logger.info("Tested agent: {}".format(agent.model_path))
    else:
        agent = construct_agent(model_type="suspected", env=env, device=device, args=args)
        agent_mode = args.suspected_agent_mode
        handler = logging.FileHandler(f"{args.env_name}fingerprint_{args.victim_agent_mode}_{args.adversary}.log")
        logger.addHandler(handler)
        logger.info("-----------------")
        logger.info("Tested agent: {}".format(agent.model_path))
        logger.info("Tested agent: {}".format(agent.model_path))

    # Initilaizations related to actor-critic models
    rhs = torch.zeros(1, 1)
    masks = torch.zeros(1, 1)
    total_scores = []
    rar = args.random_action_ratio
    vf_rar = args.vf2_random_action_ratio

    if rar > 0.0:
        logger.info("Random action return is used in test:")
    if args.vf1:
        logger.info("Visual foresight detection and recovery is used in test")
    if args.vf2:
        logger.info("VF+random recovery is used in test:")

    # load the adversarial defense, if any
    if args.vf1 or args.vf2:
        if not os.path.isfile(agent.defense_path + "acvp.pt"):
            acvp.train(args, agent)
        vf = acvp.Network(env.action_space.n).to(device)
        print(f"Load defense model from: {agent.defense_path}" + "acvp.pt")
        checkpoint = torch.load(agent.defense_path + "acvp.pt", map_location=lambda storage, loc: storage)
        vf.load_state_dict(checkpoint['model_state_dict'])
        vf.eval()
        action_one_hot = torch.zeros((env.action_space.n,)).to(device)
        vf.mean_state = checkpoint['mean_state'].to(device)
        vf.diff_threshold = get_acvp_threshold(agent_mode, args.env_name)

    #checkpoint = torch.load("Pongnoise/osfwu_none60_victim_ppo_eps0.05_v0.npy")
    #advmask = checkpoint["advmask"]

    # Environment, begin test
    for game_id in range(args.total_game_plays):
        state = env.reset()
        frame_idx_ingame = 0
        matches = 0
        distances = []

        while True:
            # Environment, render (i.e., show gameplay)
            if args.render:
                env.render()

            #if frame_idx_ingame >= 100  and frame_idx_ingame < 200:
            #    state = torch.clamp(state+advmask, 0.0, 255.0)

            # Victim, select actions
            action, action_dist = act(agent, agent_mode, masks, state, rhs, args, rar=rar)

            if args.vf1 and frame_idx_ingame > 0 :
                action, distance, match = acvp_action_correction(vf, action_dist, action)
                distances.append(distance.item())
                matches += match
            elif args.vf2 and frame_idx_ingame > 0 :
                action, distance, match = acvp2_action_correction(vf, action_dist, action, rar=vf_rar)
                distances.append(distance.item())
                matches += match

            # Victim, execute actions
            next_state, _, _, info = env.step(action)
            #if frame_idx_ingame >= 100  and frame_idx_ingame < 200:
            #    next_state = torch.clamp(next_state+advmask, 0.0, 255.0)

            # Visual foresight prediction for !next state!
            if args.vf1 or args.vf2:
                acvp_next_state_prediction(vf, action_one_hot, action, next_state)
                vf.predicted_next_action, vf.predicted_next_dist = act(agent, agent_mode, masks, vf.predicted_next_state, rhs, args)

            # Environment, prepare for next state, print if game ends
            state = next_state.clone()
            frame_idx_ingame += 1

            if 'episode' in info[0].keys():
                score = info[0]['episode']['r']
                total_scores.append(score)
                print('Game id: {}, score: {}, total number of state-action pairs: {}'.format(game_id+1,
                                                                                              info[0]['episode']['r'],
                                                                                              frame_idx_ingame))
                if args.vf1 or args.vf2:
                    distances = np.asarray(distances)
                    perc = int(len(distances)*0.95)
                    print('Visual Foresight performance {:.2f}, l1 diff mean {:.4f}, std {:.4f}, 5 percent rejection cutoff {:.4f}'.format(100*matches/frame_idx_ingame, 
                                                                                    distances.mean(), distances.std(), np.sort(distances)[perc]))
                break

    env.close()

    # Measurements, save statistics to files, log and print
    print("Purely testing score, mean:{:.2f}, std:{:.2f}".format(np.mean(total_scores), np.std(total_scores)))
    logger.info("Purely testing score, mean:{:.2f}, std:{:.2f}".format(np.mean(total_scores), np.std(total_scores)))
