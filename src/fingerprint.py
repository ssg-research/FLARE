# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import logging
import os, glob
import numpy as np

from agents.models import DQNnet
from rl_utils.utils import set_seeds
from rl_utils.atari_wrapper import make_vec_envs
from attack_utils.attacker_agent import attacker_agent
from attack_utils.load_observations import make_files_and_paths
from test import construct_agent, act
from test import acvp_action_correction, acvp2_action_correction
from test import get_acvp_threshold, acvp_next_state_prediction
from agents import action_conditional_video_prediction as acvp

def get_independent_agents(model_dir, env, args, device):

    all_agents = []
    reference_model_names = ["model1.pt", "model2.pt", "model3.pt", "model4.pt", "model5.pt"]
    files = glob.glob(model_dir + '*.pt')
    for f in files:
        model_name = (f).split('/')[-1]
        if model_name in reference_model_names:
            args.suspected_agent_path = (f)
            args.suspected_agent_mode = args.victim_agent_mode
            agent = construct_agent(model_type="independent", env=env, device=device, args=args)
            all_agents.append(agent)

    return all_agents


def generate_fingerprints(args):
    print(f"Generate {args.generate_num_masks} fingerprints.")

    logger = logging.getLogger('fingerprint_logger')
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(f"{args.env_name}fingerprint_{args.victim_agent_mode}_{args.adversary}.log")
    logger.addHandler(handler)

    device = torch.device("cuda:0" if args.cuda else "cpu")

    noise_path, training_data_path, noise_path_template = make_files_and_paths(args)

    # Create environment with random seed 
    env = make_vec_envs(args.env_name, args.seed+10, 1, None, "output/env_logs", device=device,
                        allow_early_resets=args.allow_early_resets)
    set_seeds(args, 10)

    # generate victim agent
    victim_agent = construct_agent(model_type="victim", env=env, device=device, args=args)

    # attacker is the same as victim
    args.attacker_agent_mode = args.victim_agent_mode
    adv = attacker_agent(env, victim_agent, training_data_path, noise_path, device, args)

    # Now load other independently trained reference agents to check FPR and threshold
    model_dir_path = "./output/" + args.env_name + "/" + args.victim_agent_mode + "/fingerprint/"
    independent_agents = get_independent_agents(model_dir_path, env, args, device)

    # collect data over one game to generate adv mask
    print("-------------------")
    print("Collect training data for fingerprinting...")
    rhs = torch.zeros(1, 1).to(device)
    masks = torch.zeros(1, 1).to(device)
    frame_idx_ingame = 0
    state = env.reset()

    while True:
        if args.render:
            env.render()
        victim_action, victim_action_distribution = act(victim_agent, args.victim_agent_mode,
                                            masks, state, rhs, args)

        adv.collect(state, victim_action, victim_action_distribution)


        next_state, reward, done, info = env.step(victim_action)
        state = next_state.clone()
        frame_idx_ingame += 1

        if 'episode' in info[0].keys():
            score = info[0]['episode']['r']
            print('Training episode score: {}'.format(info[0]['episode']['r'],))
            break

    print("-------------------")
    print("Generate adversarial masks with eps value {:.2f}".format(args.eps))
    logger.info("-------------------")
    logger.info("Generate adversarial masks with eps value {:.2f}".format(args.eps))

    num_masks_saved = 0
    fingerprint_list = []
    for trial in range(1000): #500
        print("Generate and test for {}-th trial".format(trial))
        action_data_clean = []
        action_data_masked = []
        dup_action_data_clean = []
        dup_action_data_masked = []
        ind_action_data_clean = []
        ind_action_data_masked = []
        result_perc_independent = []
        for i in range(len(independent_agents)):
            ind_action_data_masked.append([]) 
            ind_action_data_clean.append([]) 
        # generate universal mask
        if args.adversary == "osfwu":
            adv.generate_osfwu_masks(sampling_strategy=args.sampling_strategy, slice_id=trial)
        elif args.adversary == "cosfwu":
            adv.generate_conf_osfwu_masks(independent_agents, sampling_strategy=args.sampling_strategy, slice_id=trial)
        elif args.adversary == "uap":
            adv.generate_uap_masks()
        rhs = torch.zeros(1, 1) 
        masks = torch.zeros(1, 1)
        frame_idx_ingame = 0
        adv_min_fooling_rate = 0.8

        while True:
            if args.render:
                env.render()

            # perturb the state with the universal mask 
            state_adv, _ = adv.attack(state, frame_idx_ingame, victim_agent)
            # get actions from both victim and proxy agent (copy of the victim)
            action_c, _ = act(victim_agent, args.victim_agent_mode, masks, state, rhs, args)
            dup_action_c, _ = act(adv.proxy_agent, args.victim_agent_mode, masks, state, rhs, args)
            action_data_clean.append(action_c.item())
            dup_action_data_clean.append(dup_action_c.item())  

            action, _ = act(victim_agent, args.victim_agent_mode, masks, state_adv, rhs, args)
            dup_action, _ = act(adv.proxy_agent, args.victim_agent_mode, masks, state_adv, rhs, args)
            action_data_masked.append(action.item())
            dup_action_data_masked.append(dup_action.item())        

            for idx, ind_agent in enumerate(independent_agents):
                ind_action_c, _ = act(ind_agent, args.victim_agent_mode, masks, state, rhs, args)
                ind_action_data_clean[idx].append(ind_action_c.item())
                ind_action, _ = act(ind_agent, args.victim_agent_mode, masks, state_adv, rhs, args)
                ind_action_data_masked[idx].append(ind_action.item())
            
            next_state, _, _, info = env.step(action_c)

            state = next_state.clone()
            frame_idx_ingame += 1

            if 'episode' in info[0].keys():
                score = info[0]['episode']['r']
                print('Return: {}'.format(score))
                break

        result_perc_copy = 1.0 - np.sum(np.asarray(action_data_masked) == np.asarray(action_data_clean))/len(action_data_masked)
        print("Effectiveness of adversarial mask on victim: {:.3f}".format(result_perc_copy))

        for idx in range(len(ind_action_data_masked)):
            indices = np.where(np.asarray(ind_action_data_clean[idx]) == np.asarray(action_data_clean))[0]
            result_perc = np.sum(np.asarray(ind_action_data_masked[idx])[indices] == np.asarray(action_data_masked)[indices])/len(indices)
            print("Action agreement (aa) on victim vs independent model (On same original action) {}: {:.3f}".format(idx+1, result_perc))   
            result_perc_independent.append(result_perc)    

        if any(torch.equal(tensor, adv.advmask) for tensor in fingerprint_list) == False:
            if result_perc_copy >= adv_min_fooling_rate and all([(1.0-i)*result_perc_copy > args.nts for i in result_perc_independent]):
                print("Fingerprint {} computed in trial {}, with a fooling rate of {:.3f}".format(num_masks_saved+1, trial+1, adv.fooling_rate))
                print(f"Save fingerprint to {noise_path_template.format(num_masks_saved)}.")
                logger.info("{} fingerprint {}: computed in trial {}, with a fooling rate of {:.3f}".format(args.adversary, num_masks_saved+1, trial+1, adv.fooling_rate))
                logger.info("{} fingerprint {}: adversarial effectiveness {:.3f}".format(args.adversary, num_masks_saved+1, result_perc_copy))
                logger.info("{} fingerprint {}: aa copy {:.3f}, aa independent {:3f}".format(args.adversary, num_masks_saved+1, result_perc_copy, np.mean(result_perc_independent)))
                
                # save i-th fingerprint
                torch.save({
                    'iteration': trial+1,
                    'advmask': adv.advmask,
                    'aa_copy': result_perc_copy,
                    "aa_ind": np.asarray(result_perc_independent).mean()
                }, noise_path_template.format(num_masks_saved))

                num_masks_saved += 1
                fingerprint_list.append(adv.advmask)
        else:
            print("Mask is equal to at least one mask in the list")

        if num_masks_saved >= args.generate_num_masks:
            break

    env.close()
    print(f"Generated {num_masks_saved} masks.")



def verify_indiv_agent(args):
    print("-------------------")
    print("Verify a suspected agent with previously saved fingerprints...")

    logger = logging.getLogger('fingerprint_logger')
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(f"{args.env_name}fingerprint_{args.victim_agent_mode}_{args.adversary}.log")
    logger.addHandler(handler)

    device = torch.device("cuda:0" if args.cuda else "cpu")
    _, _, noise_path_template = make_files_and_paths(args)

    # Crreate test environment with random seed+1000
    env = make_vec_envs(args.env_name, args.seed + 1000, 1, None, "output/env_logs", device=device,
                        allow_early_resets=args.allow_early_resets)
    set_seeds(args, 1000)

    # generate victim agent
    victim_agent = construct_agent(model_type="victim", env=env, device=device, args=args)
    logger.info("-----------------")
    logger.info("Victim agent is: {}".format(victim_agent.model_path))

    # generate suspected agent
    suspected_agent = construct_agent(model_type="suspected", env=env, device=device, args=args)
    logger.info("Suspect agent: {}".format(suspected_agent.model_path))

    files = glob.glob(noise_path_template.replace("{}", "*"))
    n_detection_games = args.generate_num_masks

    fingerprint_list = []
    for f in files:
        checkpoint = torch.load(f, map_location=lambda storage, loc: storage)
        fingerprint_list.append(checkpoint["advmask"])

    #n_detection_games =  len(fingerprint_list)
    suspected_action_data = []
    original_action_data = []
    result_percentages = []
    decision_results = []
    scores = []
    for i in range(n_detection_games):
        suspected_action_data.append([])
        original_action_data.append([])

    # Adaptive evasion strategy 1: return random actions instead of maximizing return
    rar = args.random_action_ratio
    # Adaptive evasion strategy 2 and 3: return actions corrected by the visual foresight (VF) module
    if args.vf1 or args.vf2:
        if not os.path.isfile(suspected_agent.defense_path + "acvp.pt"):
            acvp.train(args, suspected_agent)
        vf = acvp.Network(env.action_space.n).to(device)
        print(f"Load model from: {suspected_agent.defense_path}" + "acvp.pt")
        checkpoint = torch.load(suspected_agent.defense_path + "acvp.pt", map_location=lambda storage, loc: storage)
        vf.load_state_dict(checkpoint['model_state_dict'])
        vf.eval()
        action_one_hot = torch.zeros((env.action_space.n,)).to(device)
        vf.mean_state = checkpoint['mean_state'].to(device)
        vf.diff_threshold = get_acvp_threshold(args.suspected_agent_mode, args.env_name)
    # Adaptive evasion strategy 3: return close to max action with ratio and VF module
    vf_rar = args.vf2_random_action_ratio

    if rar > 0.0:
        logger.info("Random action return is used for evasion:")
    if args.vf1:
        logger.info("Visual foresight detection and recovery is used for evasion")
    if args.vf2:
        logger.info("Visual foresight detection and random recovery is used for evasion")

    # collect average number of states in a null game, without verification
    state= env.reset()
    num_total_states = 0
    masks = torch.zeros(1, 1)
    rhs = torch.zeros(1, 1)
    while True:
        if args.render:
            env.render()
        action, _ = act(suspected_agent, args.suspected_agent_mode,masks, state, rhs, args)
        next_state, _, _, info = env.step(action)
        state = next_state.clone()
        num_total_states += 1
        if 'episode' in info[0].keys():
            score = info[0]['episode']['r']
            print('Null episode score: {}'.format(info[0]['episode']['r'],))
            break

    # Verify suspected policy using fingerprints
    for fingerprint_id in range(n_detection_games):
        # Load individual fingerprint using fingerprint_id
        fingerprint = fingerprint_list[fingerprint_id].to(device)
        masks = torch.zeros(1, 1)
        rhs = torch.zeros(1, 1)
        state_id = 0

        clean_aa = 0
        adv_aa = 0
        adv_aa_other = 0
        adversary_success = 0
        adversary_success_other = 0

        # calculate a start state for fingerprint verification for i-th fingerprint randomly
        if args.ver_slength < int(num_total_states/2):
            start_state = np.random.randint(0, int(num_total_states/2)-args.ver_slength)
        else:
            start_state = 0
            args.ver_slength = int(num_total_states/2)

        while True:
            if args.render:
                env.render()
        
            if state_id >= start_state and state_id < (start_state + args.ver_slength): 
                # add fingerprint aka adversarial mask to the input state
                clean_suspected_action, _ = act(suspected_agent, args.suspected_agent_mode, masks, state, rhs, rar=rar)
                clean_expected_action, _ = act(victim_agent, args.victim_agent_mode, masks, state, rhs)
                state = torch.clamp(state+fingerprint, 0.0, 255.0)
                suspected_action, suspected_action_dist = act(suspected_agent, args.suspected_agent_mode, masks, state, rhs, rar=rar)

                # action correction via VF
                if args.vf1 and state_id > 0 :
                    suspected_action, _, _ = acvp_action_correction(vf, suspected_action_dist, suspected_action)
                elif args.vf2 and state_id > 0 :
                    suspected_action, _, _ = acvp2_action_correction(vf, suspected_action_dist, suspected_action)

                # verifier gets the expected action using its own agent 
                expected_action, _ = act(victim_agent, args.victim_agent_mode, masks, state, rhs)
                if clean_suspected_action == clean_expected_action:
                    clean_aa += 1
                    if clean_expected_action != expected_action:
                        adversary_success += 1
                        if suspected_action == expected_action:
                            adv_aa += 1
                else:
                    if clean_expected_action != expected_action:
                        adversary_success_other += 1
                        if suspected_action == expected_action:
                            adv_aa_other += 1
                # verifier collects actions on this time window
                suspected_action_data[fingerprint_id].append(suspected_action.item())
                original_action_data[fingerprint_id].append(expected_action.item())
            else:
                suspected_action, suspected_action_dist = act(suspected_agent, args.suspected_agent_mode, masks, state, rhs, args, rar=rar)
                # action correction via VF
                if args.vf1 and state_id > 0 :
                    suspected_action, _, _ = acvp_action_correction(vf, suspected_action_dist, suspected_action)
                elif args.vf2 and state_id > 0 :
                    suspected_action, _, _ = acvp2_action_correction(vf, suspected_action_dist, suspected_action, rar=vf_rar)

            next_state, _, _, info = env.step(suspected_action)
            # VF module predicts the next_state, but it might be a noisy one, be careful
            if args.vf1 or args.vf2:
                if state_id >= start_state and state_id < (start_state + args.ver_slength): 
                    acvp_next_state_prediction(vf, action_one_hot, suspected_action, torch.clamp(next_state+fingerprint, 0.0, 255.0))
                else:
                    acvp_next_state_prediction(vf, action_one_hot, suspected_action, next_state)
                vf.predicted_next_action, vf.predicted_next_dist = act(suspected_agent, args.suspected_agent_mode, masks, vf.predicted_next_state, rhs, args)

            state = next_state.clone()
            state_id += 1

            if 'episode' in info[0].keys():
                delta = np.finfo(float).eps
                score = info[0]['episode']['r']
                print('Verification episode: {}, Score: {}, at total number of {} states with {}'.format(fingerprint_id+1, info[0]['episode']['r'], state_id, args.adversary))
                print("From {} actions, {} of them originally matches, fingerprint changes {:.2f} of them"
                      " and from those, the action agreement when fingerprint is present is {:.2f} ".format(len(original_action_data[fingerprint_id]), clean_aa, 
                                                                                                           adversary_success/(clean_aa+delta), 
                                                                                                           float(adv_aa)/float(adversary_success+delta)))
                print("From {} actions, {} of them does not originally match at all, but fingerprint changes {:.2f} of them "
                      " and from those, the action agreement when fingerprint is present is {:.2f} ".format(len(original_action_data[fingerprint_id]), 
                                                                                                           args.ver_slength-clean_aa, 
                                                                                                           adversary_success_other/(args.ver_slength-clean_aa+delta), 
                                                                                                           float(adv_aa_other)/float(adversary_success_other+delta)))
                logger.info("From {} actions, {} of them originally matches, fingerprint changes {:.2f} of them "
                      " and from those, the action agreement when fingerprint is present is {:.2f} ".format(len(original_action_data[fingerprint_id]), clean_aa,
                                                                                                           adversary_success/(clean_aa+delta), 
                                                                                                           float(adv_aa)/float(adversary_success+delta)))
                logger.info("From {} actions, {} of them does not originally match at all, but fingerprint changes {:.2f} of them "
                      " and from those, the action agreement when fingerprint is present is {:.2f} ".format(len(original_action_data[fingerprint_id]), 
                                                                                                           args.ver_slength-clean_aa, 
                                                                                                           adversary_success_other/(args.ver_slength-clean_aa+delta), 
                                                                                                           float(adv_aa_other)/float(adversary_success_other+delta)))
                scores.append(score)
                break

    for idx in range(len(original_action_data)):
        # if there is no mask is applied due to the short length of the episode, write result prec as 0.0, they are clearly independently trained policies. 
        if len(original_action_data[idx]) == 0:
            result_perc = 0.0
        else:
            result_perc = np.sum(np.asarray(original_action_data[idx]) == np.asarray(suspected_action_data[idx]))/len(original_action_data[idx])
        print("Action agreement (aa) on victim vs suspected model on mask {}: {:.2f}".format(idx+1, result_perc))     
        if result_perc > args.decision_threshold:
            decision_results.append("Stolen")
        else:
            decision_results.append("Independent")
        result_percentages.append(result_perc)

    
    s_vote_count = decision_results.count("Stolen")
    i_vote_count = decision_results.count("Independent")
    decision_percentage = (float(decision_results.count("Stolen")) * 100 )/ len(decision_results)
    print("Verification summary on {} states with {}:".format(args.ver_slength, args.adversary))
    print("Score of the suspected agent during verification, mean:{:.2f}, std:{:.2f}".format(np.mean(scores), np.std(scores)))
    print("Action agreement results on {} masks, mean: {:.3f}, std: {:.3f}".format(len(result_percentages), np.mean(result_percentages), np.std(result_percentages)))
    print("Stolen vote count: {}, Independent vote count: {}".format(s_vote_count, i_vote_count))
    print("Stolen by {} percentage confidence".format(decision_percentage))
    logger.info("Verification summary on {} states with {}:".format(args.ver_slength, args.adversary))
    logger.info("Score of the suspected agent during verification, mean:{:.2f}, std:{:.2f}".format(np.mean(scores), np.std(scores)))
    logger.info("Action agreement results on {} masks, mean: {:.3f}, std: {:.3f}".format(len(result_percentages), np.mean(result_percentages), np.std(result_percentages)))
    logger.info("Stolen vote count: {}, Independent vote count: {}".format(s_vote_count, i_vote_count))
    logger.info("Stolen by {} percentage confidence".format(decision_percentage))


def fingerprint(args):
    if args.generate_fingerprint:
        generate_fingerprints(args)
    else:
        verify_indiv_agent(args)
