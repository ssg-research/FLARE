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
