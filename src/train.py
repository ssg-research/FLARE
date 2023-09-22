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
import gym
import time
import torch
import random
import logging
import numpy as np
from torch import nn
from datetime import datetime
from collections import deque
import torch.nn.utils.prune as prune

import agents.action_conditional_video_prediction as acvp
from agents.dqn_agent import dqn_agent
from agents.a2c_agent import a2c_agent
from agents.ppo_agent import ppo_agent
from agents.models import DQNnet
from agents.storage import RolloutStorage
from rl_utils.atari_wrapper import make_vec_envs
from rl_utils.utils import set_seeds, EpsilonScheduler
from test import construct_agent, act
from ibp import network_bounds, sequential_bounds

def _compute_robust_loss(curr_model, target_model, data, epsilon, kappa, gamma, device, args, epsilon_end=None):
    state, action, reward, next_state, done = data

    # change everyting to cuda 
    #state, action, reward, next_state, done = state.cuda(), action.cuda(), reward.cuda(), next_state.cuda(), done.cuda()

    #adjust for the substracting mean in model.forward by making the model automatically substract the mean
    with torch.no_grad():
        curr_model.action_value.weight[:] -= torch.mean(curr_model.action_value.weight, dim=0, keepdim=True)
        curr_model.action_value.bias[:] -= torch.mean(curr_model.action_value.bias, dim=0, keepdim=True)

    q_values      = curr_model(state)
    value = curr_model.value(state)
    next_q_values = curr_model(next_state)
    target_next_q_values  = target_model(next_state) 

    q_value       = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = target_next_q_values.gather(1, torch.argmax(next_q_values, 1, keepdim=True)).squeeze(1)
    expected_q_value = reward.squeeze(1) + gamma * next_q_value * (1 - done)
    
    standard_loss = torch.min((q_value - expected_q_value.detach()).pow(2), torch.abs(q_value - expected_q_value.detach()))
    
    i_upper, i_lower = network_bounds(curr_model, state, epsilon)
    upper, lower = sequential_bounds(curr_model, i_upper, i_lower)

    upper += value.detach()
    lower += value.detach()

    onehot_labels = torch.zeros(upper.shape).to(device)
    onehot_labels[range(state.shape[0]), action] = 1

    #if args.loss_fn == "approach_1":
    #    upper_diff = (upper - expected_q_value.detach().unsqueeze(1))*onehot_labels + (upper - q_values)*(1-onehot_labels) 
    #    lower_diff = (lower - expected_q_value.detach().unsqueeze(1))*onehot_labels + (lower - q_values)*(1-onehot_labels) 
    #    wc_diff = torch.max(torch.abs(upper_diff), torch.abs(lower_diff))
    #    worst_case_loss = torch.sum(torch.min(wc_diff.pow(2), wc_diff), dim=1).mean()
    
    # approach 2 is implemented for DQN, since it was reported to give better performance
    #calculate how much worse each action is than the action taken
    q_diff = torch.max(torch.zeros([1]).to(device), 
            (q_values.gather(1, action.unsqueeze(1)).detach()-q_values.detach()))
    overlap = torch.max(upper - lower.gather(1, action.unsqueeze(1)) + q_diff.detach()/2, torch.zeros([1]).to(device))
    worst_case_loss = torch.sum(q_diff*overlap, dim=1).mean()+1e-4*torch.ones([1]).to(device)
    
    standard_loss = standard_loss.mean()
    loss = (kappa*(standard_loss)+(1-kappa)*(worst_case_loss))

    return loss, standard_loss, worst_case_loss

def train(args):
    device = torch.device("cuda:0" if args.cuda else "cpu")
    env = make_vec_envs(args.env_name, args.seed, 1, args.gamma, 'output/env_logs', device, allow_early_resets=args.allow_early_resets)
    set_seeds(args)
    if args.victim_agent_mode == 'dqn':
        dqn_train(args, env, device)
    elif args.victim_agent_mode == 'a2c' or args.victim_agent_mode == 'ppo':
        env = make_vec_envs(args.env_name, args.seed, args.num_processes, args.gamma, 'output/env_logs', device, False)
        set_seeds(args)
        policy_train(args, env, device)  


def modify(args):
    device = torch.device("cuda:0" if args.cuda else "cpu")
    set_seeds(args, 20)
    logger = logging.getLogger('fingerprint_logger')
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(f"{args.env_name}fingerprint_{args.victim_agent_mode}_{args.adversary}.log")
    logger.addHandler(handler)
    print("-------------------")

    # change the random seed from the original training seed
    if args.victim_agent_mode == "dqn":
        if args.game_mode == "finetune":
            logger.info("Finetuning is used as model modification")
            env = make_vec_envs(args.env_name, args.seed+20, 1, args.gamma, 'output/env_logs', device, allow_early_resets=args.allow_early_resets)
            agent = construct_agent(model_type="victim", env=env, device=device, args=args)
            dqn_finetune(agent, env, device, args)
        elif args.game_mode == "fineprune":
            logger.info("Pruning+Finetuning is used as model modification")
            for level in [0.25, 0.5, 0.75, 0.9]:
                env = make_vec_envs(args.env_name, args.seed+20, 1, args.gamma, 'output/env_logs', device, allow_early_resets=args.allow_early_resets)
                agent = construct_agent(model_type="victim", env=env, device=device, args=args)
                dqn_finetune(agent, env, device, args, pruning=True, plevel = level)
    else:
        if args.game_mode == "finetune":
            logger.info("Finetuning is used as model modification")
            env = make_vec_envs(args.env_name, args.seed+20, args.num_processes, args.gamma, 'output/env_logs', device, False)
            agent = construct_agent(model_type="victim", env=env, device=device, args=args)
            policy_finetune(agent, env, device, args)
        elif args.game_mode == "fineprune":
            logger.info("Pruning+Finetuning is used as model modification")
            for level in [0.25, 0.5, 0.75, 0.9]:
                env = make_vec_envs(args.env_name, args.seed+20, args.num_processes, args.gamma, 'output/env_logs', device, False)
                set_seeds(args, 20)
                agent = construct_agent(model_type="victim", env=env, device=device, args=args)
                policy_finetune(agent, env, device, args, pruning=True, plevel= level)

def adversarial_train(args):
    device = torch.device("cuda:0" if args.cuda else "cpu")
    set_seeds(args, 20)
    logger = logging.getLogger('fingerprint_logger')
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(f"{args.env_name}fingerprint_{args.victim_agent_mode}_{args.adversary}.log")
    logger.addHandler(handler)
    print("-------------------")
    logger.info("RADIAL is used for adversarial training")

    if args.victim_agent_mode == "dqn":
        env = make_vec_envs(args.env_name, args.seed+20, 1, args.gamma, 'output/env_logs', device=device, allow_early_resets=args.allow_early_resets)
        current_agent = construct_agent(model_type="victim", env=env, device=device, args=args)
        target_agent = construct_agent(model_type="victim", env=env, device=device, args=args)
        dqn_radial(current_agent, target_agent, env, device, logger, args)
    else:
        print("RADIAL adversarial training currently only supports DQN agents ...")

def dqn_train(args, env, device, agent=None, advmask=None):
    model_name = "model_original.pt"
    agent = dqn_agent(model_name, args.victim_agent_mode, env, device, args)

    # the episode reward
    episode_rewards = deque(maxlen=100)
    obs = env.reset()
    td_loss = 0.0
    timestep = 0
    start = time.time()
    # start to learn 
    for timestep in range(args.total_timesteps): 
        explore_eps = agent.exploration_schedule.get_value(timestep)   
        # select actions
        action, _ = agent.select_action(obs, explore_eps)
        # excute actions
        next_obs, reward, done, infos = env.step(action)
        if args.render:
            env.render()
        # trying to append the samples
        agent.remember(obs, action, np.sign(reward), next_obs, done)
        obs = next_obs.clone()
        # add the rewards
        #episode_reward.add_rewards(reward)
        td_loss =agent.update_agent(timestep, advmask)
        for info in infos:
            if 'episode' in info.keys():
                episode_rewards.append(info['episode']['r'])
                if timestep % args.display_interval == 0 and len(episode_rewards) > 1:
                    end = time.time()
                    print("Updates {}, num timesteps {}, \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                        .format(timestep, args.total_timesteps,
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards)))
        # save for every interval-th episode or for the last epoch
        if (timestep % (args.dqn_save_interval) == 0) and args.save_dir != "":
            #torch.save(agent.net.state_dict(), agent.model_path + model_name)
            print("Save model to" + agent.model_path)
            torch.save({
            'model_state_dict': agent.net.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            }, agent.model_path)

    if args.save_old_model:
        print(f"Save model to {agent.new_model_path + agent.new_model_name}.")
        torch.save(agent.net.state_dict(), agent.new_model_path + agent.new_model_name)

    # close the environment
    env.close()

def dqn_radial(current_agent, target_agent, env, device, logger, args):
    start_time = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    # some default args taken from RADIAL_RL main.py
    total_frames = 4500000 # divide it by four 
    replay_initial = 5000 # how many frames of experience to collect before starting to learn, divide by four
    batch_size = 128
    updates_per_frame = 32  # how many gradient updates per new frame
    attack_epsilon_schedule = 4000000 # the frame by which to reach final perturbation
    attack_epsilon_end = 1.0/255.0
    kappa_end = 0.5 # final value of the variable controlling importance of standard loss
    worse_bound = True # worst case loss uses bound that is further away from mean
    save_max = True # save model on every test run high score matched or bested
    smoothed = True # whether to use linear attack epsilon schedule instead of default smoothed one
    constant_kappa = True # whether to use a linear kappa schedule instead of default constant kappa
    kappa_end = 0.8 # final value of the variable controlling importance of standard loss 
    #gamma = 0.99 # discount factor for rewards

    #linearly decrease epsilon from 1 to epsilon end over epsilon decay steps
    epsilon_start = 1.0
    exp_epsilon_decay = 1.0
    decay_zero = False
    exp_epsilon_end = 0.01
    def action_epsilon_by_frame(frame_idx):
        if frame_idx <= exp_epsilon_decay or not decay_zero:
            return (exp_epsilon_end + max(0, 1-frame_idx/exp_epsilon_decay)*(epsilon_start-exp_epsilon_end))
        else:
            return max(0, (total_frames-frame_idx) / (total_frames-exp_epsilon_decay))*(exp_epsilon_end)
        
    #old_timestep = current_agent.exploration_schedule.total_timesteps
    #current_agent.exploration_schedule.total_timesteps = old_timestep + total_frames
    # cuda or cpu for training
    gpu_id = 0
    if device.type == "cuda":
        device = torch.device('cuda:{}'.format(0))
    else:
        device = torch.device('cpu')
    replay_buffer = ReplayBuffer(20000, device)
    optimizer = torch.optim.Adam(current_agent.net.parameters(), lr=0.000125, amsgrad=False, eps=1e-8)
    if smoothed:
        attack_eps_scheduler = EpsilonScheduler("smoothed", replay_initial, attack_epsilon_schedule, 0, 
                                                attack_epsilon_end, attack_epsilon_schedule)        

    start = time.time()
    losses = []
    standard_losses = []
    worst_case_losses = []
    all_rewards = []
    worst_case_rewards = []
    #initialize as a large negative number to save first
    best_avg = float("-inf")
    episode_reward = 0
    training_games = 0
    
    state = env.reset()
    for frame_idx in range(1, total_frames + 1):
        action_epsilon = action_epsilon_by_frame(frame_idx)
        #action_epsilon = 0.01 * (1 - (float(frame_idx) / total_frames))
        #action_epsilon = 0.01 * (1.0 - (float(frame_idx + old_timestep) / float(old_timestep + total_frames)))
        action, _ = current_agent.select_action(state, action_epsilon)
        next_state, reward, done, info = env.step(action)
        episode_reward += reward

        # scale rewards between -1 and 1
        reward     = torch.clamp(reward, min=-1, max=1).to(device)  
        done       = torch.FloatTensor(done).to(device)  
        action     = action.to(device)  
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state

        if frame_idx%100000==0:
            if smoothed:
                attack_epsilon = attack_eps_scheduler.get_eps(0, frame_idx)
            else:
                attack_epsilon = lin_coeff*attack_epsilon_end
            test_rewards = []
            for i in range(10):
                #test_reward = 0
                while True:
                    action, _ = current_agent.select_action(state)
                    next_state, reward, done, info = env.step(action)
                    #test_reward += reward  
                    state = next_state
                    if 'episode' in info[0].keys():  
                        state = env.reset()    
                        test_reward =  info[0]['episode']['r']
                        test_rewards.append(test_reward)
                        break
            avg_reward = np.mean(np.asarray(test_rewards))
            print("Steps: {}, {} games used for training, avg test reward: {}, Time taken {:.3f}".format(frame_idx, training_games, 
                                                                                                            avg_reward.item(), time.time()-start)) 
            logger.info("Steps: {}, {} games used for training, avg test reward: {}, Time taken {:.3f}".format(frame_idx, training_games, 
                                                                                                                  avg_reward.item(), time.time()-start)) 

            if save_max and avg_reward>=best_avg and len(losses)>0 :
                best_avg = avg_reward
                file_name = "robust_model_"+ str(start_time) + "_best.pt"
                path_to_folder = os.path.split(current_agent.model_path)[0]
                current_agent.model_path = os.path.join(path_to_folder, file_name)
                print("Save current best model to " + current_agent.model_path)
                logger.info("Save current best model to " + current_agent.model_path)
                torch.save({
                'model_state_dict': current_agent.net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, current_agent.model_path)
                print("Current best model saved at {}-th frame".format(frame_idx))
                logger.info("Current best model saved at {}-th frame".format(frame_idx))
            
        elif frame_idx%10000==0:
            #test_reward = 0
            while True:
                action, _ = current_agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                state = next_state
                if 'episode' in info[0].keys():  
                    state = env.reset() 
                    test_reward = info[0]['episode']['r']
                    print("Steps: {}, test reward: {}, Time taken {:.3f}".format(frame_idx, test_reward, time.time()-start)) 
                    logger.info("Steps: {}, test reward: {}, Time taken {:.3f}".format(frame_idx, test_reward, time.time()-start)) 
                    break

        if done and 'episode' in info[0].keys():
            # current episode ends, collect cumulative reward, and reset environment
            all_rewards.append(episode_reward)
            episode_reward = 0
            state = env.reset()
            training_games += 1

        if len(replay_buffer) > replay_initial and frame_idx%(batch_size/updates_per_frame)==0:
            
            init_coeff = (frame_idx - replay_initial +1)/min(attack_epsilon_schedule, total_frames)
            #clip between 0 and 1
            lin_coeff = max(min(1, init_coeff), 0)
            
            if smoothed:
                attack_epsilon = attack_eps_scheduler.get_eps(0, frame_idx)
            else:
                attack_epsilon = lin_coeff*attack_epsilon_end
            
            if constant_kappa:
                kappa = kappa_end
            else:
                kappa = (1-lin_coeff)*1 + lin_coeff*kappa_end

            data = replay_buffer.sample(args.batch_size)
            loss, standard_loss, worst_case_loss = _compute_robust_loss(current_agent.net, target_agent.net, data, attack_epsilon,
                                                                     kappa, args.gamma, device, args, attack_epsilon_end)
               
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.data.item())
            standard_losses.append(standard_loss.data.item())
            worst_case_losses.append(worst_case_loss.data.item())


        if frame_idx % (1000*(batch_size/updates_per_frame)) == 0:
            target_agent.net.load_state_dict(current_agent.net.state_dict()) 

    # save final model
    file_name = "robust_model" + "_last.pt"
    path_to_folder = os.path.split(current_agent.model_path)[0]
    current_agent.model_path = os.path.join(path_to_folder, file_name)
    print("Save robust model to " + current_agent.model_path)
    logger.info("Save robust model to " + current_agent.model_path)
    torch.save({
    'model_state_dict': current_agent.net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    }, current_agent.model_path)
    print("Robust model saved at {}-th frame".format(total_frames))
    logger.info("Robust model saved at {}-th frame".format(total_frames))

def dqn_finetune(agent, env, device, args, pruning=False, plevel=0.0):
    checkpoint = torch.load(agent.model_path)
    # load the saved checkpoint for optimizer, if any
    if 'optimizer_state_dict' in checkpoint:
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # else, start from a lower learning rate to finetune without losing performance
    else:
        for g in agent.optimizer.param_groups:
            g['lr'] = g['lr']/100.0
    print("Optimizer stats")
    print(agent.optimizer)

    num_updated_games = 0
    old_episode_reward_len = 0
    args.dqn_save_interval = 100000
    #old_timestep = agent.exploration_schedule.total_timesteps
    #agent.exploration_schedule.total_timesteps += old_timestep
    # the episode reward
    episode_rewards = deque(maxlen=100)
    recorded_rewards = []
    state = env.reset()

    # if pruning is True, first prune weights and then fine-tune
    if pruning == True:
        parameters_to_prune = (
            (agent.net.cnn_layer.conv1, "weight"),
            (agent.net.cnn_layer.conv2, "weight"),
            (agent.net.cnn_layer.conv3, "weight"),
            (agent.net.action_fc, "weight"),
            (agent.net.state_value_fc, "weight"),
            (agent.net.action_value, "weight"),
            (agent.net.state_value, "weight"),
        )

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=plevel,
        )

    if pruning == True:
        # remove the pruning before saving to make pruning permanent
        prune.remove(agent.net.cnn_layer.conv1, "weight")
        prune.remove(agent.net.cnn_layer.conv2, "weight")
        prune.remove(agent.net.cnn_layer.conv3, "weight")
        prune.remove(agent.net.action_fc, "weight")
        prune.remove(agent.net.state_value_fc, "weight")
        prune.remove(agent.net.action_value, "weight")
        prune.remove(agent.net.state_value, "weight")

    # start to learn 
    for timestep in range(1, agent.exploration_schedule.total_timesteps): 
        if num_updated_games >= 210:
            env.close()
            break

        #explore_eps = agent.exploration_schedule.get_value(timestep)   
        #explore_eps = 0.01 * (1 - (float(timestep) / agent.exploration_schedule.total_timesteps))
        explore_eps = 0.01
        # select actions
        action, _ = agent.select_action(state, explore_eps)
        # excute actions
        next_state, reward, done, infos = env.step(action)
        if args.render:
            env.render()
        # trying to append the samples
        agent.remember(state, action, np.sign(reward), next_state, done)
        state = next_state.clone()
        # add the rewards
        #episode_reward.add_rewards(reward)
        td_loss =agent.update_agent(timestep)

        for info in infos:
            if 'episode' in info.keys():
                episode_rewards.append(info['episode']['r'])
                recorded_rewards.append(info['episode']['r'])

        if len(recorded_rewards) > num_updated_games and len(recorded_rewards) > old_episode_reward_len:
            if num_updated_games % 50 == 0: #10
                # if pruning is True, then save model at only specific num of updates, no need to save all fine-pruned models necessarily
                if pruning == True:
                    if num_updated_games in [100, 200]: #num_updated_games in [0, 20, 40, 50, 100, 150, 180, 200]:
                        path_to_folder = os.path.split(agent.model_path)[0]
                        file_name = 'pruned_model' + str(num_updated_games) + '-prate-' + str(int(plevel*100.0)) + '.pt'
                        agent.model_path = os.path.join(path_to_folder, file_name)
                        print("Save model to" + agent.model_path)
                        torch.save({
                        'model_state_dict': agent.net.state_dict(),
                        'optimizer_state_dict': agent.optimizer.state_dict(),
                        }, agent.model_path)
                        print("Model saved at {}-th game, number of updated games {}".format(len(recorded_rewards), num_updated_games))    
                else:
                    path_to_folder = os.path.split(agent.model_path)[0]
                    file_name = 'finetuned_model'+ str(num_updated_games) + '.pt'
                    agent.model_path = os.path.join(path_to_folder, file_name)
                    print("Save model to " + agent.model_path)
                    torch.save({
                    'model_state_dict': agent.net.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    }, agent.model_path)
                    print("Model saved at {}-th game, number of updated games {}".format(len(recorded_rewards), num_updated_games))
            num_updated_games +=1
            old_episode_reward_len = len(recorded_rewards)
            print(
                "Finetune updates {}, \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(timestep, len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards)))

    # close the environment
    env.close()

def policy_train(args, envs, device, agent=None, advmask=None):
    #TODO: check adversarial training
    victim = True
    if args.victim_agent_mode == 'a2c':
        model_name = "model_original.pt"
        agent = a2c_agent(model_name, args.victim_agent_mode, envs, device, args)
    else:
        model_name = "model_original.pt"
        agent = ppo_agent(model_name, args.victim_agent_mode, envs, device, args)

    rollouts = RolloutStorage(args.policy_num_steps, args.num_processes,
                            (4,84,84), envs.action_space,1)
    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device) 

    episode_rewards = deque(maxlen=100)

    start = time.time()
    num_updates = int(args.total_timesteps) // args.policy_num_steps // args.num_processes
    for j in range(num_updates):

        for step in range(args.policy_num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = agent.select_action(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step], deterministic=args.deterministic_policy)


            # Observe reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = agent.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.gamma, use_proper_time_limits=False)

        value_loss, action_loss, dist_entropy = agent.update_agent(j, rollouts=rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % (args.policy_save_interval) == 0
                or j == num_updates - 1) and args.save_dir != "":
            #torch.save(agent.net.state_dict(), agent.model_path + model_name) 
            print("Save model to " + agent.model_path)
            torch.save({
            'model_state_dict': agent.net.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            }, agent.model_path)

        if j % args.policy_log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.policy_num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))
    if args.save_old_model:
        print(f"Save model to {agent.new_model_path + agent.new_model_name}.")
        torch.save(agent.net.state_dict(), agent.new_model_path + agent.new_model_name)

    envs.close()   


def policy_finetune(agent, env, device, args, pruning=False, plevel=0.0):
    checkpoint = torch.load(agent.model_path)
    # load the saved checkpoint for optimizer, if any
    if 'optimizer_state_dict' in checkpoint:
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # else, start from a lower learning rate to finetune without losing performance
    else:
        for g in agent.optimizer.param_groups:
            g['lr'] = g['lr']/100.0
    print("Optimizer stats")
    print(agent.optimizer)

    num_updated_games = 0
    old_episode_reward_len = 0
    rollouts = RolloutStorage(args.policy_num_steps, args.num_processes,
                            (4,84,84), env.action_space,1)
    state = env.reset()
    rollouts.obs[0].copy_(state)
    rollouts.to(device) 
    recurrent_hidden_states = torch.zeros(1, 1)
    masks = torch.zeros(1, 1)
    episode_rewards = deque(maxlen=100)
    recorded_rewards = []
    num_updates = int(args.total_timesteps) // args.policy_num_steps // args.num_processes

    # if pruning is True, first prune weights and then fine-tune
    if pruning == True:
        parameters_to_prune = (
            (agent.net.main[0], 'weight'),
            (agent.net.main[2], 'weight'),
            (agent.net.main[7], 'weight'),
            (agent.net.critic_linear, 'weight'),
            (agent.net.dist.linear, 'weight'),
        )

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=plevel,
        )
    if pruning == True:
        # remove the pruning before saving to make pruning permanent
        prune.remove(agent.net.main[0],"weight")
        prune.remove(agent.net.main[2],"weight")
        prune.remove(agent.net.main[7],"weight")
        prune.remove(agent.net.critic_linear, 'weight')
        prune.remove(agent.net.dist.linear, 'weight')


    for j in range(num_updates):
        if num_updated_games >= 210:
            env.close()
            break
        for step in range(args.policy_num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = agent.select_action(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step], deterministic=args.deterministic_policy)

            # Observe reward and next state
            state, reward, done, infos = env.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    recorded_rewards.append(info['episode']['r'])

            # If done then clean the history of states.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                for info in infos])
            rollouts.insert(state, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = agent.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.gamma, use_proper_time_limits=False)
        value_loss, action_loss, dist_entropy = agent.update_agent(j, rollouts=rollouts)
        rollouts.after_update()
        total_num_steps = (j + 1) * args.num_processes * args.policy_num_steps

        if len(recorded_rewards) > num_updated_games and len(recorded_rewards) > old_episode_reward_len:
            if num_updated_games % 50 == 0:
                # if pruning is True, then save model at only specific num of updates, no need to save all fine-pruned models necessarily
                if pruning == True:
                    if num_updated_games in [100, 200]: #[0, 20, 50, 100, 150, 180, 200]:
                        path_to_folder = os.path.split(agent.model_path)[0]
                        file_name = 'pruned_model' + str(num_updated_games) + '-prate-' + str(int(plevel*100.0)) + '.pt'
                        agent.model_path = os.path.join(path_to_folder, file_name)
                        print("Save model to" + agent.model_path)
                        torch.save({
                        'model_state_dict': agent.net.state_dict(),
                        'optimizer_state_dict': agent.optimizer.state_dict(),
                        }, agent.model_path)
                        print("Model saved at {}-th game, number of updated games {}".format(len(recorded_rewards), num_updated_games))     
                else:
                    path_to_folder = os.path.split(agent.model_path)[0]
                    file_name = 'finetuned_model'+ str(num_updated_games) + '.pt'
                    agent.model_path = os.path.join(path_to_folder, file_name)
                    print("Save model to" + agent.model_path)
                    torch.save({
                    'model_state_dict': agent.net.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    }, agent.model_path)
                    print("Model saved at {}-th game, number of updated games {}".format(len(recorded_rewards), num_updated_games))
            num_updated_games +=1
            old_episode_reward_len = len(recorded_rewards)
            print(
                "\nFinetune updates {}, num timesteps {}, \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}"
                .format(j, total_num_steps, len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))
    env.close() 

class ReplayBuffer(object):
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.device = device
        self.buffer = []
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (torch.cat(state, dim=0), torch.cat(action, dim=0), torch.cat(reward, dim=0), 
                torch.cat(next_state, dim =0), torch.cat(done, dim=0))
        
    
    def __len__(self):
        return len(self.buffer)



