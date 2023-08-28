# FLARE: Fingerprinting Deep Reinforcement Learning Agents using Universal Adversarial Masks
This repo contains the code to reproduce experimental results in the paper *FLARE: Fingerprinting Deep Reinforcement Learning Agents using Universal Adversarial Masks*. The work will appear in the proceedings of ACSAC 2023. You can access the ArXiv pre-print from [here](https://arxiv.org/abs/2307.14751).

## Setup
This code has been tested to work with GPU (GeForce GTX 1060 6GB) and CUDA=10.1. In order to replicate the same results in paper, we suggest using the same hardware setup. However, the code has been also tested with both GPU and CPU instances in Amazon Sagemaker Studio Lab. We recommend using conda to run and replicate experiments smoothly. Dependencies and OpenAI baselines can be installed by following the steps below:
```
conda create -n flare python=3.7
conda activate flare
pip3 install -r requirements.txt
cd src
git clone https://github.com/openai/baselines.git
cd baselines
pip3 install -e .
```
## Training and Testing Your Own Policies
Victim models used for our experimental setup are given under **src/output/** directory. However, you can also train your own models and evaluate their performance by measuring the average return over test episodes. 
Training your own agent is simple:

`
python main.py --game-mode train --env-name $GAME --victim-agent-mode $VICTIM_AGENT_MODE --seed $SEED --cuda
`
  > --game-mode: train, test or fingerprint, default is train. \
  > --victim-agent-mode: the type of policy, a2c, dqn or ppo, default is set to dqn. \
  > --env-name: name of the game, Pong or MsPacman or any other available ALE game, default is Pong. \
  > --seed: the number to generate the random starting state in gameplays (or episodes), default is set to 123. \
  > --cuda: If set, the training will happen using GPU, otherwise in CPU. If there is no cuda available, --cuda option will return error. 

This will generate the trained policy in folder ./output/$GAME$/$VICTIM_AGENT_MODE/train/model_original.pt. You can check main.py to further modify the training hyperparamaters, e.g., learning rate, total time steps, entropy term coefficient in a2c and ppo, etc. Beware that training your own model might take some from, ranging from couple of hours to a day. 

To evalute your agents' performance:

`
python main.py --game-mode test --env-name $GAME --victim-agent-mode $VICTIM_AGENT_MODE --seed $SEED --victim-agent-path ./the/folder/for/policy.pt --cuda
`
  > --victim-agent-path: the path to victim agent, if none is given, the default path is `./output/$GAME$/$VICTIM_AGENT_MODE/test/model_original.pt` 
 
## Fingerprinting Generation and Verification

FLARE works in two steps:
1. Generate fingerprint
2. Verify a suspected model

### Fingerprint Generation:

`
python main.py --game-mode fingerprint --env-name $GAME --adversary $ADVERSARY --victim-agent-mode $VICTIM_AGENT_MODE --generate-fingerprint --eps 0.05 --generate-num-masks 10 -cuda 
`
  > --adversary: the type of adversarial example generation method, none, random (random gaussion noise), cosfwu (conferrable universal adversarial masks), oswf (universal adversarial masks by [Pan et al](https://arxiv.org/abs/1907.09470)), uap (universal adversarial perturbations by [Moosavi-Dezfooli et al.](https://arxiv.org/abs/1610.08401)). This method is used to generate fingerprints for the victim model ```model_original.pt``` stored in `output/$GAME/$VICTIM_AGENT_MODE/fingerprint`.\
  > --generate-fingerprint: this flag should be set only during fingerprint generation phase. \
  > --eps: the maximum amount of l_infinity norm on the adversarial mask, default is set to 0.05. \
  > --generate-num-masks: the number of fingerprints to be generated, default is set to 10. \
  > --nts: the minimum non-transferability score for an adversarial mask to be included in the fingerprint list, default is set to 0.5.\
  > --cuda: If set, the training will happen using GPU, otherwise in CPU. If there is no cuda available, --cuda option will return error.

You can check main.py to further modify the hyperparamaters, (e.g., number of training frames used in the cofwu/osfwu algorithm, number of episodes to collect D_flare). Please remember, for any victim model, you need to train 5 more independent models to generate the fingerprint list. We do not provide independent models used in the paper, but you can download fingerprints computed in our experimental setup from [here](https://drive.google.com/file/d/1I3r4v7MFE2Tq-1xer7FhAgjFohNFV9QX/view?usp=sharing) if you want to reproduce the verification results for modified policies. 

### Fingerprint Verification:
`
  python main.py --game-mode fingerprint --env-name $GAME --adversary $ADVERSARY --victim-agent-mode $VICTIM_AGENT_MODE --eps 0.05 --suspected-agent-mode $SUSPECTED_AGENT_MODE
                    --suspected-agent-path ./the/folder/for/suspected/policy.pt --ver-slength 40 --cuda
`
  > --suspected-agent-mode: the type of suspected agent's policy, a2c, dqn or ppo, default is dqn. \
  > --suspected-agent-path: the path to suspected agent. This should be an existing path to a suspected agent to succesfully tun the verification algorithm. You should also make sure that the suspected agent policy is correct while loading it. \
  > --ver-slength: the window size (i.e., the total number of states) used in the verification, default is set to 40.

## Model Modification and Evasion Attacks

In this repository, we provide fine-tuning and pruning as model modification attacks. We also provide random action return, adversarial example detection and recovery with [visual foresight](https://arxiv.org/abs/1710.00814) (VF1), and visual foresight with a suboptimal action (VF2) as evasion strategies. For fine-tuning and pruning, you need to change --game-mode to finetune and prune, respectively. You also need to generate finetune or fineprune folder under `output/$GAME/$VICTIM_AGENT_MODE/finetune` (or  `output/$GAME/$VICTIM_AGENT_MODE/fineprune`), and move the victi model `model_original.pt` to these folders in order to load the correct model. For VF1 and VF2, you need to train visual foresight modules, and then use one of these options during fingerprint verification: 
  > --random-action-ratio $RATIO: random action return with a ratio $RATIO, default is set to 0.0, maximum is 1.0. \
  > --vf1: If set, visual foresight with correct action recovery setup is initiated. \
  > --vf2 --vf2-random-action-ratio $RATIO: If set, visual foresight with suboptimal actions is initiated with a given random action ratio.

For running different attack strategies (as well as fingerprint generation and verification), please check the bash script `src/complete_experiments.sh`.  

## Licence
This project is licensed under Apache License Version 2.0. By using, reproducing or distributing to the project, you agree to the license and copyright terms therein and release your version under these terms.

## Acknowledgements
We built our environments with [OpenAI's Atari Gym](https://github.com/gsurma/atari). We also want to thank other researchers out there for making their repository publicly available. Our DQN implemention is inspired from [here](https://github.com/williamd4112/RL-Adventure), Our A2C and PPO implementations are based on [here](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail). Our UAP implementation is based on [this](https://github.com/ferjad/Universal_Adversarial_Perturbation_pytorch) repository. 
