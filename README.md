# Image-Gen-and-Enhancement-using-SDXL-SwinIR

# Stable Diffusion XL + SwinIR Image Generation & Enhancement Pipeline

This project integrates **Stable Diffusion XL (SDXL)** — a cutting-edge text-to-image generation model — with **SwinIR**, a powerful image restoration and super-resolution model, to generate high-quality, high-resolution images from text prompts.

---

## Table of Contents

- [Overview](#overview)  
- [Project Structure](#project-structure)  
- [Requirements](#requirements)  
- [Usage](#usage)  
  - [Inference](#inference)  
  - [Training SwinIR](#training-swinir)  
- [Customization](#customization)  
- [Notes](#notes)  
- [References](#references)  
- [License](#license)  
- [Contact](#contact)  

---

## Overview

- **Stable Diffusion XL (SDXL):** Generates images from text prompts.
- **SwinIR:** Enhances the quality of generated images through image restoration and super-resolution.

The pipeline first generates an image from text with SDXL, then uses SwinIR to enhance image resolution and details.

---

## Project Structure

sd_swinir_project/
├── models/
│ ├── stable_diffusion_xl.py # SDXL architecture & generation code
│ └── swinir.py # SwinIR model definition
├── utils/
│ ├── scheduler.py # (Optional) LR schedulers
│ └── metrics.py # PSNR, SSIM, FID metrics
├── train_combined.py # Fine-tune SwinIR on SDXL outputs
├── inference.py # Combined SDXL + SwinIR inference script
├── config.yaml # Configuration file
├── requirements.txt # Python dependencies
└── README.md # This file


---

## Requirements

- Python 3.8 or higher
- PyTorch 1.12+
- torchvision
- timm
- Pillow
- numpy


## Usage
Inference
Generate and enhance an image from a text prompt:

python inference.py

Output image saved as combined_output.png.

## Training SwinIR on SDXL Outputs
Fine-tune SwinIR to better enhance SDXL-generated images:
python train_combined.py
Fine-tuned weights saved as swinir_finetuned.pth.






