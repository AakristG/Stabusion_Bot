# Sable Diffusion with PyTorch

This is a simple implementation of a Stable Diffusion bot that is used as a reference and sample to generate images for AdFusion (https://github.com/AakristG/AdFusion)

This was more a research on how to build on machine learning and code for the blog!

Before running this make sure to follow the steps before:

## Set up your environment

1. Download Jupyter Notebook and run it with a GPU (CPU is highly not recommended)
2. Create your environment and download all the required libraries: torch, numpy, and tqdm.

## Download weights and tokenizer files:

1. Download `vocab.json` and `merges.txt` from https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main/tokenizer and save them in the `data` folder
2. Download `v1-5-pruned-emaonly.ckpt` from https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main and save it in the `data` folder

## Reference
A huge thanks to https://github.com/hkproj/pytorch-stable-diffusion !