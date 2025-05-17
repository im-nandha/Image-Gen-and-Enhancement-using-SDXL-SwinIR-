from PIL import Image
import torch
from utils.metrics import compute_ssim, compute_psnr, compute_fid
import os
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

real_dir = "real_images/"
fake_dir = "generated_images/"

real_imgs = [transform(Image.open(os.path.join(real_dir, f))) for f in os.listdir(real_dir)]
fake_imgs = [transform(Image.open(os.path.join(fake_dir, f))) for f in os.listdir(fake_dir)]

# Compute metrics
fid = compute_fid(real_imgs, fake_imgs)
psnr_vals = [compute_psnr(r, f) for r, f in zip(real_imgs, fake_imgs)]
ssim_vals = [compute_ssim(r, f) for r, f in zip(real_imgs, fake_imgs)]

print(f"FID: {fid:.2f}")
print(f"PSNR: {sum(psnr_vals)/len(psnr_vals):.2f}")
print(f"SSIM: {sum(ssim_vals)/len(ssim_vals):.4f}")
