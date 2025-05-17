import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def compute_psnr(img1, img2):
    img1 = img1.cpu().numpy().transpose(1,2,0)
    img2 = img2.cpu().numpy().transpose(1,2,0)
    return psnr(img1, img2)

def compute_ssim(img1, img2):
    img1 = img1.cpu().numpy().transpose(1,2,0)
    img2 = img2.cpu().numpy().transpose(1,2,0)
    return ssim(img1, img2, multichannel=True)