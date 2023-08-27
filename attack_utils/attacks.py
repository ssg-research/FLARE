import copy
import torch
import random
import os,sys
import pickle
import numpy as np
import torch.nn as nn
import time
import math

from attack_utils.load_observations import *
from attack_utils.deepfool import deepfool

from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import tqdm
import glob

from inspect import getargspec

def project_lp(v, xi, p):
    if p==2:
        pass
    elif p == np.inf:
        v=np.sign(v)*np.minimum(abs(v),xi)
    else:
        raise ValueError("Values of a different from 2 and Inf are currently not surpported...")
    return v

def forward(agent, cur_obs):
    # If agent takes two arguments, then it's DQN agent
    # ... else, it's A2C agent
    if len(getargspec(agent.net.forward).args) == 2:
        action_values = agent.net(cur_obs)
    else:
        _, actor_features, _ =  agent.net(cur_obs, torch.zeros(1,1), torch.zeros(1, 1))#agent.net(cur_obs, torch.zeros(1,1), torch.zeros(1, 1))
        dist = agent.net.dist(actor_features)
        action_numbers = torch.linspace(0, agent.env.action_space.n - 1, steps=agent.env.action_space.n).to(agent.device)
        action_values = [dist.log_probs(ac.to(agent.device).reshape(1, 1)) for ac in action_numbers]
        action_values = torch.cat(action_values).reshape(1, len(action_numbers))
        #change action values to softmax and return that (Huang et. al FGSM into DRL)
    #return torch.nn.functional.softmax(action_values, dim=1)
    return action_values

def deepfool_attack(agent, obs, xi, device):
    # Deepfool untargeted adversarial attack on single states
    v, _, _, _, _ = deepfool(obs.squeeze(), agent.net, num_classes=agent.env.action_space.n, overshoot=0.2, max_iter=50)
    v = project_lp(v, xi, np.inf)
    v = torch.tensor(v, dtype=torch.float32).to(device)
    obs_adv = torch.clamp(obs+v, 0.0, 255.0)
    return obs_adv


def generate_osfw(agent, eps, collected_states, device, n_training_frames, sampling_strategy='none', slice_id=0):

    if n_training_frames == -1:
        n_training_frames = len(collected_states)

    num_obs_trn = len(collected_states)
    q_variances = torch.zeros(num_obs_trn).to(agent.device)
    grad_samples = torch.zeros(num_obs_trn, 4, 84, 84).to(agent.device)

    for idx, state_vect in enumerate(collected_states):
        state = state_vect.clone().detach().requires_grad_(True)
        action_vect = forward(agent, state.unsqueeze(0))
        target = torch.argmax(action_vect)
        loss = torch.nn.functional.cross_entropy(action_vect, target.unsqueeze(0))
        loss.backward()
        fgsm_grad = state.grad.detach()
        q_variances[idx] = torch.var(action_vect)
        grad_samples[idx] = fgsm_grad.clone()    

    if sampling_strategy == 'random':
        idxs = random.sample(range(len(grad_samples)), n_training_frames)
        grad_samples_for_mask = grad_samples[idxs]
    elif sampling_strategy == 'slice':
        idxs = []
        if num_obs_trn >= n_training_frames*(slice_id+1):
            grad_samples_for_mask = grad_samples[n_training_frames*slice_id:n_training_frames*(slice_id+1)]
            print(grad_samples_for_mask.size())
            print(n_training_frames*slice_id)
            print(n_training_frames*(slice_id+1))
            idxs.extend(range(n_training_frames*slice_id, n_training_frames*(slice_id+1)))
        else:
            grad_samples_for_mask = grad_samples[n_training_frames*slice_id:num_obs_trn]
            idxs.extend(range(n_training_frames*slice_id, num_obs_trn))
    else:
        num_samples = n_training_frames if num_obs_trn >= n_training_frames else num_obs_trn
        grad_samples_for_mask = grad_samples[:num_samples]
        idxs = []
        idxs.extend(range(0,num_samples))

    print(f"Using {len(grad_samples_for_mask)} states to generate osfw mask.")

    # take the average of the gradients with size eps
    v = torch.mean(grad_samples_for_mask, 0)
    v = eps * 256.0 * v.sign()
    #v = v.sign() * torch.min(abs(v), torch.ones(v.shape).to(agent.device)*eps) * 256.0

    #if num_obs_trn < n_training_frames:
    #    print("Did not collect enough samples to construct universal perturbation using {} frames.".format(n_training_frames))
    #    print("Construct adversarial perturbation using {} frames instead...".format(num_obs_trn))

    with torch.no_grad():
        clip_min = 0.0
        clip_max = 255.0
        fooling_rate_batch = []
        test_sample_size = 0
        for id, inputs in enumerate(collected_states):
            if id not in idxs:
                inputs = inputs.to(device)
                outputs = forward(agent, inputs.unsqueeze(0))
                _, predicted_clean = outputs.max(1)
                v_torch = v.clone().detach().unsqueeze(0).to(device)
                inputs = torch.clamp(inputs + v_torch, clip_min, clip_max)
                outputs = forward(agent, inputs)
                _, predicted_adv = outputs.max(1)
                fooling_rate_batch.append(float(torch.sum(predicted_clean.reshape(len(predicted_clean),)!= predicted_adv))/len(inputs))
                torch.cuda.empty_cache()
                test_sample_size += 1

        print("Fooling rate on remaining {} states: {:.3f}".format(test_sample_size, np.mean(fooling_rate_batch)))

    return v, np.mean(fooling_rate_batch), idxs

def generate_conferrable_osfw(agent, independent_agents, eps, collected_states, device, n_training_frames, sampling_strategy='none', slice_id=0):

    if n_training_frames == -1:
        n_training_frames = len(collected_states)

    num_obs_trn = len(collected_states)
    q_variances = torch.zeros(num_obs_trn).to(agent.device)
    grad_samples = torch.zeros(num_obs_trn, 4, 84, 84).to(agent.device)

    for idx, state_vect in enumerate(collected_states):
        state = state_vect.clone().detach().requires_grad_(True)
        action_vect = forward(agent, state.unsqueeze(0))
        target = torch.argmax(action_vect)
        loss = torch.nn.functional.cross_entropy(action_vect, target.unsqueeze(0))
        ind_loss = 0
        ind_match = 0
        for ind_agent in independent_agents:
            ind_action_vect = forward(ind_agent, state.unsqueeze(0))
            if torch.argmax(ind_action_vect).unsqueeze(0) == target.unsqueeze(0):
                ind_loss += torch.nn.functional.cross_entropy(ind_action_vect, torch.argmax(ind_action_vect).unsqueeze(0))
                ind_match += 1
        if ind_match > 0:
            loss = loss - ind_loss/ind_match
        loss.backward()
        fgsm_grad = state.grad.detach()
        q_variances[idx] = torch.var(action_vect)
        grad_samples[idx] = fgsm_grad.clone()    

    if sampling_strategy == 'random':
        idxs = random.sample(range(len(grad_samples)), n_training_frames)
        grad_samples_for_mask = grad_samples[idxs]
    elif sampling_strategy == 'slice':
        idxs = []
        if num_obs_trn >= n_training_frames*(slice_id+1):
            grad_samples_for_mask = grad_samples[n_training_frames*slice_id:n_training_frames*(slice_id+1)]
            print(grad_samples_for_mask.size())
            print(n_training_frames*slice_id)
            print(n_training_frames*(slice_id+1))
            idxs.extend(range(n_training_frames*slice_id, n_training_frames*(slice_id+1)))
        else:
            grad_samples_for_mask = grad_samples[n_training_frames*slice_id:num_obs_trn]
            idxs.extend(range(n_training_frames*slice_id, num_obs_trn))
    else:
        num_samples = n_training_frames if num_obs_trn >= n_training_frames else num_obs_trn
        grad_samples_for_mask = grad_samples[:num_samples]
        idxs = []
        idxs.extend(range(0,num_samples))

    print(f"Using {len(grad_samples_for_mask)} states to generate osfw mask.")

    # take the average of the gradients with size eps
    v = torch.mean(grad_samples_for_mask, 0)
    v = eps * 256.0 * v.sign()
    #v = v.sign() * torch.min(abs(v), torch.ones(v.shape).to(agent.device)*eps) * 256.0

    #if num_obs_trn < n_training_frames:
    #    print("Did not collect enough samples to construct universal perturbation using {} frames.".format(n_training_frames))
    #    print("Construct adversarial perturbation using {} frames instead...".format(num_obs_trn))

    with torch.no_grad():
        clip_min = 0.0
        clip_max = 255.0
        fooling_rate_batch = []
        test_sample_size = 0
        for id, inputs in enumerate(collected_states):
            if id not in idxs:
                inputs = inputs.to(device)
                outputs = forward(agent, inputs.unsqueeze(0))
                _, predicted_clean = outputs.max(1)
                v_torch = v.clone().detach().unsqueeze(0).to(device)
                inputs = torch.clamp(inputs + v_torch, clip_min, clip_max)
                outputs = forward(agent, inputs)
                _, predicted_adv = outputs.max(1)
                fooling_rate_batch.append(float(torch.sum(predicted_clean.reshape(len(predicted_clean),)!= predicted_adv))/len(inputs))
                torch.cuda.empty_cache()
                test_sample_size += 1

        print("Fooling rate on remaining {} states: {:.3f}".format(test_sample_size, np.mean(fooling_rate_batch)))

    return v, np.mean(fooling_rate_batch), idxs


def generate_uap(agent, eps, collected_states, collected_distributions, device, n_training_states=500, delta=0.2, 
                    max_iter_uni=50, p=np.inf, overshoot=0.2, max_iter_df=50):
    # n training states was 200 for ppo and a2c

    agent.net.eval()
    agent.net = agent.net.to(device)

    # v is the universal adversarial mask initialized to zero
    v = np.zeros([4, 84, 84])
    v_torch = torch.tensor(v, dtype=torch.float32).to(device)
    fooling_rate = 0.0
    early_stop_iter = 0
    fooling_rate_max = 0.0
    iter = 0
    clip_min = 0.0
    clip_max = 255.0
    xi = clip_max*eps

    num_classes = len(collected_distributions[0])
    collected_states = torch.stack(collected_states)
    collected_distributions = torch.stack(collected_distributions)

    n_training_states = int(4*len(collected_states)/5)
    idxs = random.sample(range(len(collected_states)), n_training_states)
    collected_states_train = collected_states[idxs]

    while (fooling_rate) < 1-delta and iter < max_iter_uni:
        print("Pass number ", iter)
        print("Number of observations in the current batch is {}".format(len(collected_states_train)))
        num_obs_trn = len(collected_states_train)
        order = np.arange(num_obs_trn)
        np.random.shuffle(order)
        pbar = tqdm.tqdm(order)
        for k in pbar:
            # Get label for clean image
            cur_img  = collected_states_train[k].to(device)
            r2 = torch.argmax(forward(agent, cur_img.unsqueeze(0)))
            torch.cuda.empty_cache()
            # Get label for perturbed image with v
            v_torch = torch.tensor(v, dtype=torch.float32).to(device)
            per_img = torch.clamp(cur_img + v_torch, clip_min, clip_max)
            r1 = torch.argmax(forward(agent, per_img.unsqueeze(0)))
            torch.cuda.empty_cache()
            # If labels match, then pass over Deepfool algorithm to find another perturbation value 
            if (r1 == r2):
                dr, iter_k, label, k_i, pert_image = deepfool(per_img.squeeze(0), agent.net, 
                                                                num_classes=num_classes, overshoot=overshoot, 
                                                                max_iter=max_iter_df)

                if iter_k < max_iter_df-1:
                    # Add dr to v and project to lp norm
                    v[0, :, :] += dr[0, 0, :, :]
                    v[1, :, :] += dr[0, 1, :, :]
                    v[2, :, :] += dr[0, 2, :, :]
                    v[3, :, :] += dr[0, 3, :, :]
                    v = project_lp(v, xi, p)
                    
        iter = iter + 1

        # Calculate fooling rate on the training dataset
        fooling_rate_batch = []
        with torch.no_grad():
            #batch = len(dataX)
            clip_min = 0.0
            clip_max = 255.0
            for id, inputs in enumerate(collected_states):
                if id not in idxs:
                    inputs = inputs.to(device)
                    outputs = forward(agent, inputs.unsqueeze(0))#net(inputs)
                    _, predicted_clean = outputs.max(1)
                    v_torch = torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(device)
                    inputs = torch.clamp(inputs + v_torch, clip_min, clip_max)
                    outputs = forward(agent, inputs)#net(inputs.squeeze(1))
                    _, predicted_adv = outputs.max(1)
                    fooling_rate_batch.append(float(torch.sum(predicted_clean.reshape(len(predicted_clean),)!= predicted_adv))/len(inputs))
            torch.cuda.empty_cache()

        fooling_rate = np.mean(fooling_rate_batch)
        # if the fooling rate stays the same, early stop
        if abs(fooling_rate_max - fooling_rate) <= 1e-4:
            early_stop_iter += 1
        else:
            early_stop_iter = 0
        print("FOOLING RATE: ", fooling_rate)
        if fooling_rate >= fooling_rate_max:
            early_stop_iter == 0
            print("saving new noise, max fooling rate {:.3f}, previous max fooling rate {:.3f}".format(fooling_rate, fooling_rate_max))
            fooling_rate_max = fooling_rate
            v_torch = torch.tensor(v, dtype=torch.float32).to(device)
            if (fooling_rate) >= 1-delta or iter == max_iter_uni:
                return v_torch, fooling_rate, idxs         
        if early_stop_iter == 5:
            print("early stop")
            return v_torch, fooling_rate, idxs

