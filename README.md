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
Training your own agent is simple:

`
python main.py --game-mode train --env-name $GAME --victim-agent-mode $VICTIM_AGENT_MODE --seed $SEED --cuda
`
  > --game-mode: train, test or fingerprint, default is train. \
  > --victim-agent-mode: the type of policy, a2c, dqn or ppo, default is dqn. \
  > --env-name: name of the game, Pong or MsPacman or any other available ALE game, default is Pong. \
  > --seed: the number to generate the random starting state in gameplays (or episodes), default is 123. \
  > --cuda: If set, the training will happen using GPU, otherwise in CPU. If there is no cuda available, --cuda option will return error. 

This will generate the trained policy in folder ./output/$GAME$/$VICTIM_AGENT_MODE/train/model_original.pt. You can check main.py to further modify the training hyperparamaters, e.g., learning rate, total time steps, entropy term coefficient in a2c and ppo, etc.

To evalute your policy's performance:

`
python main.py --game-mode test --env-name $GAME --victim-agent-mode $VICTIM_AGENT_MODE --seed $SEED --victim-agent-path ./the/folder/for/policy.pt --cuda
`
  > --victim-agent-path: the path to victim agent, if none is given, the default path is ./output/$GAME$/$VICTIM_AGENT_MODE/test/model_original.pt 
 
## Fingerprinting Generation and Verification

FLARE works in two steps:
1. Generate fingerprint
2. Verify a suspected model

To generate fingerprints:
```
python main.py --game-mode fingerprint --env-name $GAME --adversary $ADVERSARY --victim-agent-mode $VICTIM_AGENT_MODE --generate-fingerprint --eps 0.05 --cuda 
```
This generates fingerprint using the ```model_original.pt``` model stored in `output/$GAME/$VICTIM_AGENT_MODE/fingerprint`.

## Licence
This project is licensed under Apache License Version 2.0. By using, reproducing or distributing to the project, you agree to the license and copyright terms therein and release your version under these terms.

## Acknowledgements
We built our environments with [OpenAI's Atari Gym](https://github.com/gsurma/atari). We also want to thank other researchers out there for making their repository publicly available. Our DQN implemention is inspired from [here](https://github.com/williamd4112/RL-Adventure), Our A2C and PPO implementations are based on [here](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail). Our UAP implementation is based on [this](https://github.com/ferjad/Universal_Adversarial_Perturbation_pytorch) repository. 
