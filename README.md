# Image Restoration by Denoising Diffusion Models with Iteratively Preconditioned Guidance

## 📖[**Paper**](https://arxiv.org/pdf/2312.16519.pdf)


Tomer Garber, [Tom Tirer](https://scholar.google.com/citations?user=_6bZV20AAAAJ)

The Open University of Israel and Bar-Ilan University  

# Introduction
This repository contains the code release for *Image Restoration by Denoising Diffusion Models with Iteratively
Preconditioned Guidance* (***DDPG***).

## Abstract
Training deep neural networks has become a common approach for addressing image restoration problems. An alternative for
training a “task-specific” network for each observation model is to use pretrained deep denoisers for imposing only the
signal’s prior within iterative algorithms, without additional training. 

Recently, a sampling-based variant of this approach has become popular with the rise of diffusion/score-based generative models. 
Using denoisers for general purpose restoration requires guiding the iterations to ensure agreement of the signal with the observations.

In low-noise settings, guidance that is based on backprojection (BP) has been shown to be a promising strategy
(used recently also under the names “pseudoinverse” or “range/null-space” guidance). However, the presence of noise in 
the observations hinders the gains from this approach. In this paper, we propose a novel guidance technique, 
based on preconditioning that allows traversing from BP-based guidance to least squares based guidance along the restoration scheme.

The proposed approach is robust to noise while still having much simpler implementation than alternative methods 
(e.g., it does not require SVD or a large number of iterations). We use it within both an optimization scheme and a 
sampling-based scheme, and demonstrate its advantages over existing methods for image deblurring and super-resolution.

## Supported degradations

1. Super-Resolution (Bicubic)
2. Gaussian Deblurring
3. Motion Deblurring

# Setup
## Installation
### Docker
The repository contains [Dockerfile](Dockerfile), in order to use it:
1. Clone this repository
2. cd to `DDPG` directory
3. Run:
```bash
docker build .
```
4. Run the new image

In order to run *IDPG* instead of *DDPG*, You can swap the `CMD` in the Docker file.
### Pip
## Pre-Trained Models
## Datasets
## Evaluation

TODO: 
table of contents
fill above
add images
datasets
replace intro with abstract
Note that it based on DDNM
results table
citations



