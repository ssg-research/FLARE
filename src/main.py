# Authors: Buse G. A. Tekgul
# Copyright 2020 Secure Systems Group, Aalto University, https://ssg.aalto.fi
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import atari_py

from test import test
from fingerprint import fingerprint
from train import train, modify, adversarial_train

def get_args():
    parse = argparse.ArgumentParser()
    available_games = list((''.join(x.capitalize() or '_' for x in word.split('_')) for word in atari_py.list_games()))
    parse.add_argument('--env-name', default='Pong', help='Choose from available games: ' + str(available_games) + ". Default is 'breakout'.")
    parse.add_argument('--env-type', type=str, default='atari', help='the environment type')
    parse.add_argument('--game-mode', type=str, default='train', help="Choose from available modes: train, test, fingerprint, finetune, fineprune, advtrain, imitate. Default is 'train'.")
    parse.add_argument('--victim-agent-mode', default='dqn', help="Choose from available RL algorithms: dqn, a2c, ppo, Default is 'dqn'.")
    parse.add_argument("--victim-agent-path", type=str, default="", help="the model path fort the victim agent.")
    parse.add_argument('--seed', type=int, default=123, help='the random seeds')
    parse.add_argument('--num-processes',type=int, default=16, help='how many training CPU processes to use (default: 16)')
    parse.add_argument('--cuda', action='store_true',  help='if use the gpu')
    parse.add_argument('--grad-norm-clipping', type=float, default=10, help='the gradient clipping')
    parse.add_argument('--total-timesteps', type=int, default=int(4e7), help='the total timesteps to train network')   #int(2e8)
    parse.add_argument('--total-game-plays', type=int, default=int(10), help='the total number of independent game plays in test time')
    parse.add_argument('--save-dir', type=str, default='saved_models/', help='the folder to save models')
    parse.add_argument('--display-interval', type=int, default=5, help='the display interval')
    parse.add_argument('--render', type=bool, default=False, help='render environment')
    parse.add_argument('--save-frames', type=bool, default=True, help= 'save frames for attack or no attack conditions')
    parse.add_argument('--allow-early-resets', action="store_true", help= 'allows early resets in game. !!In Freeway, you should allow early resets.')
    parse.add_argument('--save-old-model', action='store_true', help='Do not overwrite previous models, create a new directory to store current model.')
    parse.add_argument('--robust', action='store_true', help='choose the victim model that is adversarially trained')

    # DQN related arguments
    parse.add_argument('--batch-size', type=int, default=32, help='the batch size of updating')
    parse.add_argument('--lr', type=float, default=1e-4, help='learning rate of the algorithm')
    parse.add_argument('--init-ratio', type=float, default=1, help='the initial exploration ratio')
    parse.add_argument('--exploration_fraction', type=float, default=0.1, help='decide how many steps to do the exploration')
    parse.add_argument('--final-ratio', type=float, default=0.01, help='the final exploration ratio')
    parse.add_argument('--buffer-size', type=int, default=10000, help='the size of the buffer')
    parse.add_argument('--learning-starts', type=int, default=10000, help='the frames start to learn')
    parse.add_argument('--train-freq', type=int, default=4, help='the frequency to update the network')
    parse.add_argument('--target-network-update-freq', type=int, default=1000, help='the frequency to update the target network')
    parse.add_argument('--use-double-net', action='store_true', help='use double dqn to train the agent')
    parse.add_argument('--use-dueling', action='store_false', help='use dueling to train the agent')
    parse.add_argument('--gamma', type=float, default=0.99, help='the discount factor of RL')
    parse.add_argument('--dqn-save-interval',type=int,default=10000,help='save interval, one save per n updates (default: 100)')
    parse.add_argument('--deterministic-policy', action='store_true', help='Have the policy model act in deterministic way.')

    # Policy methods related arguments
    parse.add_argument('--policy-value-loss-coef',type=float,default=0.5,help='value loss coefficient (default: 0.5)')
    parse.add_argument('--policy-entropy-coef', type=float, default=0.01, help='entropy term coefficient (default: 0.01)')
    parse.add_argument('--policy-max-grad-norm',type=float, default=0.5, help='max norm of gradients (default: 0.5)')
    parse.add_argument('--policy-num-steps', type=int, default=5,help='number of forward steps in Policy agents (default: 5)')
    parse.add_argument('--policy-log-interval',type=int,default=10,help='log interval, one log per n updates (default: 10)')
    parse.add_argument('--policy-save-interval',type=int,default=100,help='save interval, one save per n updates (default: 100)')
    parse.add_argument('--use-gae', type=bool, default=False, help='use generalized advantage estimation')
    parse.add_argument('--gae-lambda', type=float, default=0.95, help='gae lambda parameter (default: 0.95)')

    # A2C related parameters
    parse.add_argument('--a2c-lr', type=float, default=7e-4, help='a2c learning rate (default: 7e-4)')
    parse.add_argument('--a2c-eps', type=float, default=1e-5, help='RMSprop optimizer epsilon (default: 1e-5)')
    parse.add_argument('--a2c-alpha', type=float, default=0.99, help='RMSprop optimizer apha (default: 0.99)')
    # todo: deal with it.. do we still use this?
    parse.add_argument('--a2c-eval-interval', type=int, default=None, help='eval interval, one eval per n updates (default: None)')

    # PPO related arguments
    parse.add_argument('--ppo-lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parse.add_argument('--ppo-eps', type=float, default=1e-5, help='Adam optimizer epsilon (default: 1e-5)')
    parse.add_argument('--ppo-clip-param', type=float, default=0.2, help='ppo clip parameter (default: 0.2)')
    parse.add_argument('--ppo-epoch', type=int, default=4, help='number of ppo epochs (default: 4)')
    parse.add_argument('--ppo-num-mini-batch', type=int, default=32, help='number of batches for ppo (default: 32)')
    parse.add_argument('--ppo-use-clipped-value-loss', type=bool, default=True, help='PPO use cliiped value loss (default: True)')
    parse.add_argument('--use-linear-lr-decay', type=bool, default=False, help='Use linear lr decay in PPO (default: False)')

    # Fingerprint generation arguments
    parse.add_argument('--adversary', default='osfwu', help="Choose from available modes: none, random, uap, osfwu, cosfwu, Default is 'none'.")
    parse.add_argument('--attacker-agent-mode', default='dqn', help="Choose from available RL algorithms: dqn, a2c, ppo, Default is 'dqn'.")
    parse.add_argument('--attacker-game-plays', type=int, default=int(1), help='the total number of independent game plays in training time for attack')
    parse.add_argument('--eps', type=float, default=0.05, help="Epsilon bound for generating adversarial examples")
    parse.add_argument('--training-frames', type=int, default=60, help="Number of frames used to train obs_fgsm_wb. Set value to -1 to use all training frames.")
    parse.add_argument('--training-frames-type', default='none', help="For obs-fgsm-wb, take largest/smallest q variance frames. Available modes: L, S, none. Default is 'none'.")
    parse.add_argument("--generate-fingerprint", action="store_true", help="To generate fingerprint based on a source policy.")
    parse.add_argument("--plot-fingerprint", action="store_true", help="To generate fingerprint based on a source policy.")
    parse.add_argument("--generate-num_masks", type=int, default=10, help="Number of masks to generate.")
    parse.add_argument("--nts", type=float, default=float(0.5)) #0.9 for oswf, 0.5 for window size=100

    # Fingerprint verification arguments
    parse.add_argument("--suspected-agent-path", type=str, default="", help="the model path fort he suspected agent.")
    parse.add_argument('--suspected-agent-mode', default='dqn', help="Choose from available RL algorithms: dqn, a2c, ppo, Default is 'dqn'.")
    parse.add_argument("--num-action-sample", type=int, default=100, help="Number of actions sampled to test for fingerprint.")
    parse.add_argument("--num-samples", type=int, default=10, help="Number of times to sample actions from one run.")
    parse.add_argument("--decision-threshold", type=float, default=float(0.5))
    parse.add_argument("--sampling-strategy", type=str, default="random")
    parse.add_argument("--ver-slength", type=int, default=100, help="number of states used for verification")

    # Adaptive attacker strategies
    parse.add_argument("--finetune-timesteps", type=int, default=1e8)
    parse.add_argument("--random-action-ratio", type=float, default=float(0.0), help = "random actions with the ratio returned as evasion")
    parse.add_argument("--vf1", action="store_true", help="if true, ACVP + action correction is used as evasion")
    parse.add_argument("--vf2", action="store_true", help="if true, ACVP +random action is used as evasion")
    parse.add_argument("--vf2-random-action-ratio", type=float, default=float(0.0), help = "random actions with the ratio returned as evasion")
    parse.add_argument("--test-episodes", type=int, default=int(100), help='the total number observed spisodes used to train visual foresight and imitation module')
    parse.add_argument("--imit-lr", type=float, default=1e-3, help='lr for imitation learning training')
    parse.add_argument("--imit-epochs", type=int, default=100, help='Imitation learning total training epocs')

    args = parse.parse_args()

    return args


if __name__ == '__main__':
    # get arguments
    args = get_args()
    if any(args.adversary in s for s in ['none', 'random', 'uap', 'osfwu', 'cosfwu']) == False:
        print('Incorrect adversary type. Use --help')
        exit(1)
    print ("Selected environment: " + str(args.env_name).upper())
    # Based on the enviornment automatically select early resets as true
    if args.env_name == "MsPacman" or args.env_name == "Freeway":
        args.allow_early_resets = True
    print ("Victim DRL algorithm: " + str(args.victim_agent_mode).upper())
    print ("Suspected DRL algorithm: " + str(args.suspected_agent_mode).upper())
    print ("Selected game mode: " + str(args.game_mode).upper())
    print ("Random action ratio (evasion 1): " + str(args.random_action_ratio))
    print ("Visual Foresight (evasion 2): " + str(args.vf1))
    print ("Visual Foresight + random action (evasion 3): " + str(args.vf2))
    print("cuda: " + str(args.cuda))


    if args.game_mode == "train":
        train(args)
    elif args.game_mode == "test":
        test(args)
    elif args.game_mode == "fingerprint":
        print ("Adversary type: " + str(args.adversary).upper())
        print ("Epsilon bound: " + str(args.eps))
        fingerprint(args)
    elif args.game_mode == "finetune" or args.game_mode == "fineprune":
        modify(args)
    elif args.game_mode == "advtrain":
        adversarial_train(args)
    else:
        print ("Unrecognized mode. Use --help")
        exit(1)

