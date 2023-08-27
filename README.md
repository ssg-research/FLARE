# FLARE: Fingerprinting Deep Reinforcement Learning Agents using Universal Adversarial Masks
This repo contains the code to reproduce experimental results in the paper *FLARE: Fingerprinting Deep Reinforcement Learning Agents using Universal Adversarial Masks*. The work will appear in the proceedings of ACSAC 2023. You can access the ArXiv pre-print from [here](https://arxiv.org/abs/2307.14751).

## Setup
This code has been tested to work with GPU (GeForce GTX 1060 6GB) and CUDA=10.1. In order to replicate the same results in paper, we suggest using the same hardware setup. However, the code has been also tested with both GPU and CPU instances in Amazon Sagemaker Studio Lab. We recommend using conda to run and replicate experiments smoothly. Dependencies and OpenAI baselines can be installed by following the steps below:
```
conda create -n flare python=3.7
conda activate flare
pip3 install -r requirements.txt
git clone https://github.com/openai/baselines.git
cd baselines
pip3 install -e .
```

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
