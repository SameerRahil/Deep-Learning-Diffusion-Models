# Diffusion Models: DDPM, DDIM and Conditional Latent Diffusion

Project 5 of CS 437 Deep Learning, LUMS, Spring 2026. Three notebooks that build a diffusion model from the noise schedule up, speed up its sampling, and move it into a learned latent space with class conditioning.

## Parts

| Part | Notebook | What it covers |
| --- | --- | --- |
| A. DDPM from scratch | [`a_ddpm_from_scratch.ipynb`](notebooks/a_ddpm_from_scratch.ipynb) ([nbviewer](https://nbviewer.org/github/SameerRahil/Deep-Learning-Diffusion-Models/blob/main/notebooks/a_ddpm_from_scratch.ipynb), [Colab](https://colab.research.google.com/github/SameerRahil/Deep-Learning-Diffusion-Models/blob/main/notebooks/a_ddpm_from_scratch.ipynb)) | Linear and cosine noise schedules, the forward process and posterior, a U-Net with attention and sinusoidal time embeddings (4.97M parameters), the simple loss against the variational bound, and reverse sampling on MNIST. |
| B. DDIM sampling | [`b_ddim_sampling.ipynb`](notebooks/b_ddim_sampling.ipynb) ([nbviewer](https://nbviewer.org/github/SameerRahil/Deep-Learning-Diffusion-Models/blob/main/notebooks/b_ddim_sampling.ipynb), [Colab](https://colab.research.google.com/github/SameerRahil/Deep-Learning-Diffusion-Models/blob/main/notebooks/b_ddim_sampling.ipynb)) | Deterministic DDIM sampling from the trained DDPM, sampling speed against step count, and interpolation between samples in noise space. |
| C. Conditional latent diffusion | [`c_conditional_latent_diffusion.ipynb`](notebooks/c_conditional_latent_diffusion.ipynb) ([nbviewer](https://nbviewer.org/github/SameerRahil/Deep-Learning-Diffusion-Models/blob/main/notebooks/c_conditional_latent_diffusion.ipynb), [Colab](https://colab.research.google.com/github/SameerRahil/Deep-Learning-Diffusion-Models/blob/main/notebooks/c_conditional_latent_diffusion.ipynb)) | A convolutional VAE and latent normalization, a class-conditional latent U-Net (6.09M parameters) with classifier-free guidance, latent drift analysis, and AdaCorrection (adaptive offset cache correction), evaluated on FID, cache hit rate, time, PSNR and SSIM. |

GitHub's notebook preview can be slow or fail to load. If it does, the nbviewer link next to each notebook shows the same file, and the Colab link opens it ready to run.

## Results

- **DDPM:** best training loss 0.0371.
- **DDIM speed:** 10 sampling steps took 0.52 s against 32.59 s for 1,000 steps.

**Conditional latent diffusion, 500 samples (lower FID is better):**

| Configuration | FID | Cache hit rate | Time (s) |
| --- | --- | --- | --- |
| Full recompute | 39.80 | 0.00 | 464 |
| Cache only, no correction | 159.63 | 0.99 | 28 |
| AdaCorrection, gamma 0.5, lag 5 | 161.50 | 0.96 | 62 |
| AdaCorrection, gamma 1.0, lag 5 | 40.09 | 0.07 | 466 |
| AdaCorrection, gamma 2.0, lag 5 | 40.84 | 0.01 | 514 |
| AdaCorrection, gamma 1.0, lag 10 | 40.85 | 0.05 | 504 |

Reusing cached features alone was about 17 times faster but destroyed quality. Correction restored quality only when it rarely reused the cache, so in this setup speed and quality could not both be kept.

## Checkpoints

[`checkpoints/best_ddpm.pt`](checkpoints) is the trained DDPM that Part B loads (about 20 MB).

## Stack

Python, PyTorch, torch-fidelity, scikit-image, Matplotlib.

## Running the notebooks

The notebooks were run on Google Colab or Kaggle with a GPU. Open one in Colab, set the runtime to GPU, and run the cells in order. MNIST downloads through torchvision. Part B expects the Part A checkpoint, included under checkpoints/.

## About

Coursework for CS 437 Deep Learning, LUMS, Spring 2026, shared as part of my portfolio (course grade: A+). The notebooks keep the original handout text and my implementations, answers and outputs.

If you are taking this course, please do your own work.

**Sameer Rahil**, BS Computer Science, LUMS. More projects at [github.com/SameerRahil](https://github.com/SameerRahil).
