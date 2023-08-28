############################################################################
# Copyright belongs to https://github.com/tuomaso/radial_rl_v2 #
############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

def initial_bounds(x0, epsilon):
    '''
    x0 = input, b x c x h x w
    '''
    upper = x0+epsilon
    lower = x0-epsilon
    return upper, lower

def weighted_bound(layer, prev_upper, prev_lower):
    prev_mu = (prev_upper + prev_lower)/2
    prev_r = (prev_upper - prev_lower)/2
    mu = layer(prev_mu)
    if type(layer)==nn.Linear:
        r = F.linear(prev_r, torch.abs(layer.weight))
    elif type(layer)==nn.Conv2d:
        r = F.conv2d(prev_r, torch.abs(layer.weight), stride=layer.stride, padding=layer.padding)
    
    upper = mu + r
    lower = mu - r
    return upper, lower

def activation_bound(layer, prev_upper, prev_lower):
    upper = layer(prev_upper)
    lower = layer(prev_lower)
    return upper, lower

def network_bounds(model, x0, epsilon):
    '''
    get inteval bound progation upper and lower bounds for the actiavtion of a model
    
    model: a nn.Sequential module
    x0: input, b x input_shape
    epsilon: float, the linf distance bound is calculated over
    '''
    upper, lower = initial_bounds(x0/255.0, epsilon)
    # weighted bound and activation bounds for cnn layers
    upper, lower = weighted_bound(model.cnn_layer.conv1, upper, lower)
    upper = F.relu(upper)
    lower = F.relu(lower)
    upper, lower = weighted_bound(model.cnn_layer.conv2, upper, lower)
    upper = F.relu(upper)
    lower = F.relu(lower)
    upper, lower = weighted_bound(model.cnn_layer.conv3, upper, lower)
    upper = F.relu(upper)
    lower = F.relu(lower)

    # activation bound before last layers
    upper = torch.flatten(upper, start_dim=1)
    lower = torch.flatten(lower, start_dim=1)

    return upper, lower

def sequential_bounds(model, upper, lower):
    '''
    get interval bound progation upper and lower bounds for the activation of a model,
    given bounds of the input
    
    model: a nn.Sequential module
    upper: upper bound on input layer, b x input_shape
    lower: lower bound on input layer, b x input_shape
    '''
    upper, lower = weighted_bound(model.action_fc, upper, lower)
    upper = F.relu(upper)
    lower = F.relu(lower)
    upper, lower = weighted_bound(model.action_value, upper, lower)
    return upper, lower

def network_bounds_old(model, x0, epsilon, dueling):
    '''
    get inteval bound progation upper and lower bounds for the actiavtion of a model
    
    model: a nn.Sequential module
    x0: input, b x input_shape
    epsilon: float, the linf distance bound is calculated over
    '''
    upper, lower = initial_bounds(x0/255.0, epsilon)
    # weighted bound and activation bounds for cnn layers
    upper, lower = weighted_bound(model.cnn_layer.conv1, upper, lower)
    upper = F.relu(upper)
    lower = F.relu(lower)
    upper, lower = weighted_bound(model.cnn_layer.conv2, upper, lower)
    upper = F.relu(upper)
    lower = F.relu(lower)
    upper, lower = weighted_bound(model.cnn_layer.conv3, upper, lower)
    upper = F.relu(upper)
    lower = F.relu(lower)

    # activation bound before last layers
    upper = torch.flatten(upper, start_dim=1)
    lower = torch.flatten(lower, start_dim=1)

    # dueling or not
    if not dueling:
        upper, lower = weighted_bound(model.fc1, upper, lower)
        upper = F.relu(upper)
        lower = F.relu(lower)
        upper, lower = weighted_bound(model.action_value, upper, lower)
    else:
        upper_a, lower_a = weighted_bound(model.action_fc, upper, lower)
        upper_a = F.relu(upper_a)
        lower_a = F.relu(lower_a)        
        upper_a, lower_a = weighted_bound(model.action_value, upper_a, lower_a)

        upper_s, lower_s = weighted_bound(model.state_value_fc, upper, lower)
        upper_s = F.relu(upper_s)
        lower_s = F.relu(lower_s)
        upper_s, lower_s = weighted_bound(model.state_value, upper_s, lower_s)
        # action values mean
        upper_a = upper_a - torch.mean(upper_a, dim=1, keepdim=True)
        lower_a = lower_a - torch.mean(lower_a, dim=1, keepdim=True)
        # Q = V + A
        upper = upper_s + upper_a
        lower = lower_s + lower_a

    return upper, lower