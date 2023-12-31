#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Modified from https://github.com/wuyx/DeepRL_pytorch                #
#######################################################################

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import torch.optim
import os, sys
from tqdm import tqdm

import test
from rl_utils.utils import set_seeds
from rl_utils.atari_wrapper import make_vec_envs

from matplotlib import pyplot as plt

class Batcher:
    def __init__(self, batch_size, data):
        self.batch_size = batch_size
        self.data = data
        self.num_entries = len(data[0])
        self.reset()

    def reset(self):
        self.batch_start = 0
        self.batch_end = self.batch_start + self.batch_size

    def end(self):
        return self.batch_start >= self.num_entries

    def next_batch(self):
        batch = []
        for d in self.data:
            batch.append(d[self.batch_start: self.batch_end])
        self.batch_start = self.batch_end
        self.batch_end = min(self.batch_start + self.batch_size, self.num_entries)
        return batch

    def shuffle(self):
        indices = np.arange(self.num_entries)
        np.random.shuffle(indices)
        self.data = [d[indices] for d in self.data]

class Network(nn.Module):
    def __init__(self, num_actions):
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 8, 2, (1, 1))
        self.conv2 = nn.Conv2d(64, 128, 6, 2, (1, 1))
        self.conv3 = nn.Conv2d(128, 128, 7, 2, (1, 1))
        self.conv4 = nn.Conv2d(128, 128, 4, 2, (0, 0))

        self.hidden_units = 128 * 3 * 3

        self.fc5 = nn.Linear(self.hidden_units, 2048)
        self.fc_encode = nn.Linear(2048, 2048)
        self.fc_action = nn.Linear(num_actions, 2048)
        self.fc_decode = nn.Linear(2048, 2048)
        self.fc8 = nn.Linear(2048, self.hidden_units)

        self.deconv9 = nn.ConvTranspose2d(128, 128, 4, 2, (0, 0))
        self.deconv10 = nn.ConvTranspose2d(128, 128, 7, 2, (1, 1))
        self.deconv11 = nn.ConvTranspose2d(128, 64, 6, 2, (1, 1))
        self.deconv12 = nn.ConvTranspose2d(64, 1, 8, 2, (1, 1))

        self.init_weights()
        self.criterion = nn.MSELoss()
        self.opt = torch.optim.Adam(self.parameters(), 1e-4)

        # visual foresight module related stuffs 
        self.diff_threshold  = 0.01
        self.predicted_next_state = None
        self.predicted_next_dist = None
        self.predicted_next_action = None
        self.mean_state = None

    def init_weights(self):
        for layer in self.children():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(layer.weight.data)
            nn.init.constant_(layer.bias.data, 0)
        nn.init.uniform_(self.fc_encode.weight.data, -1, 1)
        nn.init.uniform_(self.fc_decode.weight.data, -1, 1)
        nn.init.uniform_(self.fc_action.weight.data, -0.1, 0.1)

    def forward(self, obs, action):
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view((-1, self.hidden_units))
        x = F.relu(self.fc5(x))
        x = self.fc_encode(x)
        action = self.fc_action(action)
        x = torch.mul(x, action)
        x = self.fc_decode(x)
        x = F.relu(self.fc8(x))
        x = x.view((-1, 128, 3, 3))
        x = F.relu(self.deconv9(x))
        x = F.relu(self.deconv10(x))
        x = F.relu(self.deconv11(x))
        x = self.deconv12(x)
        return x

    def fit(self, x, a, y):
        x = Variable(x)
        a = Variable(a)
        y = Variable(y)
        y_ = self.forward(x, a)
        loss = self.criterion(y_, y)
        self.opt.zero_grad()
        loss.backward()
        for param in self.parameters():
            param.grad.data.clamp_(-0.1, 0.1)
        self.opt.step()
        return loss

    def evaluate(self, x, a, y):
        x = Variable(x)
        a = Variable(a)
        y = Variable(y)
        y_ = self.forward(x, a)
        loss = self.criterion(y_, y)
        return np.asscalar(loss.cpu().data.numpy())

    def predict(self, x, a):
        x = Variable(x)
        a = Variable(a)
        return self.forward(x,a)

def pre_process(x, mean_obs):
    return (x - mean_obs) / 255.0

def post_process(y, mean_obs):
    return (y * 255.0 + mean_obs)

def generate_dataset(args, env, agent, training_data_path, dist=False):
    max_ep = args.test_episodes
    device = torch.device("cuda:0" if args.cuda else "cpu")
    mean_state = torch.zeros(env.reset().shape).to(device)
    # initializations related to actor-critic models
    rhs = torch.zeros(1,1)
    masks = torch.zeros(1, 1)

    for ep in range(max_ep):
        state = env.reset()  
        states = []
        actions = []
        next_states = [] 
        while True:
            if args.render:
                env.render() 
            with torch.no_grad():
                action, action_distribution = test.act(agent, args.suspected_agent_mode, masks, state, rhs, args)

            # Execute actions
            next_state, _, _, info = env.step(action)
            # append observations, actions and next observations
            states.append(state.squeeze(0).to(device))
            next_states.append(next_state.squeeze(0).to(device)) 
            if dist == False:
                action_one_hot = torch.zeros((env.action_space.n,))
                action_one_hot[action] = 1
                actions.append(action_one_hot.to(device))
            else: 
                actions.append(action_distribution)
            state = next_state.clone()
            if 'episode' in info[0].keys():
                print('Game id: {}, score: {}'.format(ep, info[0]['episode']['r']))
                torch.save({"states": torch.stack(states), "actions": torch.stack(actions), "next_states": torch.stack(next_states)}, 
                            training_data_path + str(ep)+".pt")
                break
        mean_state += torch.stack(states).mean(0)/max_ep
    torch.save({"mean_state": mean_state, "total_eps": max_ep}, agent.defense_path + "meta.pt")
    env.close()


def train(args, agent):
    print("-------------------")
    print("Train action conditional video prediction module...")

    device = torch.device("cuda:0" if args.cuda else "cpu")
    # Environment, create
    env = make_vec_envs(args.env_name, args.seed + 1001, 1, None, 'output/env_logs', device=device,
                        allow_early_resets=args.allow_early_resets)
    # Environment, set seeds
    set_seeds(args, 1001)

    num_actions = env.action_space.n
    net = Network(num_actions).to(device)
    net_path = agent.defense_path
    training_data_path = agent.defense_path + 'train_dataset/'
    if not os.path.isdir(training_data_path):
        os.makedirs(training_data_path)
        generate_dataset(args, env, agent, training_data_path)

    meta = torch.load(net_path + "meta.pt", map_location=lambda storage, loc: storage)
    episodes = meta['total_eps']
    mean_state = meta['mean_state'].mean(1).to(device)

    # take the first 0.90 of episodes as training
    # ... and rest as test data
    train_episodes = int(episodes * 0.90)
    indices_train = np.arange(train_episodes)
    iteration = 1
    for _ in range(10):
        net.train()
        np.random.shuffle(indices_train)
        for ep in indices_train:
            # load training data in batches with a batch size of 32
            info = torch.load(training_data_path + str(ep) + ".pt", map_location=lambda storage, loc: storage)
            inputs = info["states"]
            actions = info["actions"]
            targets = info["next_states"]
            batcher = Batcher(32, [inputs, actions, targets])
            batcher.shuffle()
            while not batcher.end():
                _, a, x = batcher.next_batch()
                x, a = x.to(device), a.to(device)
                x_t = pre_process(x, mean_state)
                loss = net.fit(x_t[:,:3,:,:], a, x_t[:,-1,:,:].unsqueeze(1))
                if iteration % 100 == 0:
                    print('Iteration %d, training loss %f' % (iteration, loss))
                iteration += 1

        # test
        del batcher
        del info
        net.eval()
        losses = []
        test_indices = range(train_episodes, episodes)
        ep_to_print = np.random.choice(test_indices)
        for test_ep in tqdm(test_indices):
            # load test data in batches with a batch size of 32
            info = torch.load(training_data_path + str(ep) + ".pt", map_location=lambda storage, loc: storage)
            inputs = info["states"]
            actions = info["actions"]
            targets = info["next_states"]
            test_batcher = Batcher(32, [inputs, actions, targets])
            while not test_batcher.end():
                with torch.no_grad():
                    _, a, x = test_batcher.next_batch()
                    x, a = x.to(device), a.to(device)
                    x_t = pre_process(x, mean_state)
                    losses.append(net.evaluate(x_t[:,:3,:,:], a, x_t[:,-1,:,:].unsqueeze(1)))
            if test_ep == ep_to_print:
                test_batcher.reset()
                with torch.no_grad():
                    _, a, x = test_batcher.next_batch()
                    x, a = x.to(device), a.to(device)
                    x_t = pre_process(x, mean_state)
                    y_ = post_process(net.predict(x_t[:,:3,:,:], a), mean_state)
                    plt.imsave(training_data_path + '%d.png' % (iteration), y_[30,0,:,:].cpu().detach().numpy(), cmap='gray')
                    plt.imsave(training_data_path + '%d-truth.png' % (iteration), x[30,-1,:,:].cpu().detach().numpy(), cmap='gray')
       
        print('Iteration %d, test loss %f' % (iteration, np.mean(losses)))
        torch.save({
            'mean_state': mean_state,
            'model_state_dict': net.state_dict()},  
            net_path +  "acvp.pt")
