from __future__ import division
import numpy as np
import random
import torch
from baselines.common.vec_env.vec_normalize import VecNormalize

# set random seeds for the pytorch, numpy and random
def set_seeds(args, rank=0):
    # set seeds for the numpy
    np.random.seed(args.seed + rank)
    # set seeds for the random.random
    random.seed(args.seed + rank)
    # set seeds for the pytorch
    torch.manual_seed(args.seed + rank)
    if args.cuda:
        torch.cuda.manual_seed(args.seed + rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# record the reward info of the dqn experiments
class reward_recorder:
    def __init__(self, history_length=100):
        self.history_length = history_length
        # the empty buffer to store rewards 
        self.buffer = [0.0]
        self._episode_length = 1
    
    # add rewards
    def add_rewards(self, reward):
        self.buffer[-1] += reward

    # start new episode
    def start_new_episode(self):
        if self.get_length >= self.history_length:
            self.buffer.pop(0)
        # append new one
        self.buffer.append(0.0)
        self._episode_length += 1

    # get length of buffer
    @property
    def get_length(self):
        return len(self.buffer)
    
    @property
    def mean(self):
        return np.mean(self.buffer)
    
    # get the length of total episodes
    @property 
    def num_episodes(self):
        return self._episode_length


def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

class EpsilonScheduler():
    def __init__(self, schedule_type, init_step, final_step, init_value, final_value, num_steps_per_epoch, mid_point=.25, beta=4.):
        self.schedule_type = schedule_type
        self.init_step = init_step
        self.final_step = final_step
        self.init_value = init_value
        self.final_value = final_value
        self.mid_point = mid_point
        self.beta = beta
        self.num_steps_per_epoch = num_steps_per_epoch
        assert self.final_value >= self.init_value
        assert self.final_step >= self.init_step
        assert self.beta >= 2.
        assert self.mid_point >= 0. and self.mid_point <= 1.

    def get_eps(self, epoch, step):
        if self.schedule_type == "smoothed":
            return self.smooth_schedule(epoch * self.num_steps_per_epoch + step, self.init_step, self.final_step, self.init_value, self.final_value, self.mid_point, self.beta)
        else:
            return self.linear_schedule(epoch * self.num_steps_per_epoch + step, self.init_step, self.final_step, self.init_value, self.final_value)

    # Smooth schedule that slowly morphs into a linear schedule.
    # Code is adapted from DeepMind's IBP implementation:
    # https://github.com/deepmind/interval-bound-propagation/blob/2c1a56cb0497d6f34514044877a8507c22c1bd85/interval_bound_propagation/src/utils.py#L84
    def smooth_schedule(self, step, init_step, final_step, init_value, final_value, mid_point=.25, beta=4.):
        """Smooth schedule that slowly morphs into a linear schedule."""
        assert final_value >= init_value
        assert final_step >= init_step
        assert beta >= 2.
        assert mid_point >= 0. and mid_point <= 1.
        mid_step = int((final_step - init_step) * mid_point) + init_step
        if mid_step <= init_step:
            alpha = 1.
        else:
            t = (mid_step - init_step) ** (beta - 1.)
            alpha = (final_value - init_value) / ((final_step - mid_step) * beta * t + (mid_step - init_step) * t)
        mid_value = alpha * (mid_step - init_step) ** beta + init_value
        is_ramp = float(step > init_step)
        is_linear = float(step >= mid_step)
        return (is_ramp * (
            (1. - is_linear) * (
                init_value +
                alpha * float(step - init_step) ** beta) +
            is_linear * self.linear_schedule(
                step, mid_step, final_step, mid_value, final_value)) +
                (1. - is_ramp) * init_value)

    # Linear schedule.
    # Code is adapted from DeepMind's IBP implementation:
    # https://github.com/deepmind/interval-bound-propagation/blob/2c1a56cb0497d6f34514044877a8507c22c1bd85/interval_bound_propagation/src/utils.py#L73
    def linear_schedule(self, step, init_step, final_step, init_value, final_value):
        """Linear schedule."""
        assert final_step >= init_step
        if init_step == final_step:
            return final_value
        rate = float(step - init_step) / float(final_step - init_step)
        linear_value = rate * (final_value - init_value) + init_value
        return np.clip(linear_value, min(init_value, final_value), max(init_value, final_value))

