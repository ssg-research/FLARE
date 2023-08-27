import numpy as np
from torch.autograd import Variable
import torch as torch
import copy
from torch.autograd.gradcheck import zero_gradients

from inspect import getargspec

def forward(net,x, num_classes):
    # Required variables for A2C
    recurrent_hidden_states = torch.zeros(1,1)
    masks = torch.zeros(1, 1)
    if len(getargspec(net.forward).args) == 2:
        #return torch.nn.functional.softmax(net(x), dim=1)
        return net(x)
    else:
        _, actor_features, _ = net(x, recurrent_hidden_states, masks)
        dist = net.dist(actor_features)
        action_numbers = torch.linspace(0,num_classes-1,steps=num_classes)
        complete_action_log_probs = [dist.log_probs(ac.reshape(1,1).to('cuda')) for ac in action_numbers]
        complete_action_log_probs = torch.cat(complete_action_log_probs).reshape(1, len(action_numbers))
        #return torch.nn.functional.softmax(complete_action_log_probs, dim=1)
        return complete_action_log_probs

def deepfool(image, net, num_classes, overshoot, max_iter, single_frame=False):

    """
       :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        image = image.cuda()
        net = net.cuda()

    f_image = forward(net,Variable(image[None, :, :, :], requires_grad=True), num_classes).data.cpu().numpy().flatten() 
    #net.forward(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
    I = f_image.argsort()[::-1]

    I = I[0:num_classes]
    label = I[0]

    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = Variable(pert_image[None, :], requires_grad=True)
    y = Variable(image[None, :], requires_grad=True)
    diff = x-y
    
    fs = forward(net, x, num_classes) #net.forward(x)
    k_i = label
    diffloss = torch.nn.CrossEntropyLoss()
    
    while k_i == label and loop_i < max_iter:

        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()
               

        for k in range(1, num_classes):
            zero_gradients(x)

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

            pert_k = abs(f_k)/(np.linalg.norm(w_k.flatten())+ np.finfo(float).eps)

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # convert w to a constrained version
        if single_frame:
            w_temp = np.zeros(w.shape)
            for i in range(w.shape[1]):
                w_temp[0,i,:,:] = (w[0,0,:,:] + w[0,1,:,:] + w[0,2,:,:] + w[0,3,:,:])/4.0 
            w = w_temp
        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i =  (pert+1e-4) * w / (np.linalg.norm(w)+ np.finfo(float).eps)
        r_tot = np.float32(r_tot + r_i)

        if is_cuda:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).cuda()
        else:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot)

        x = Variable(pert_image, requires_grad=True)
        fs = forward(net, x, num_classes) #net.forward(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())

        loop_i += 1
        per_tot = (1+overshoot)*r_tot

    return (1+overshoot)*r_tot, loop_i, label, k_i, pert_image
