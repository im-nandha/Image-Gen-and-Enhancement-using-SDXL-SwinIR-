import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from torchvision.models.inception import inception_v3
from torchvision.transforms import Resize, ToTensor, Normalize, Compose
from scipy.linalg import sqrtm
from PIL import Image

def compute_ssim(img1, img2):
    img1 = img1.permute(1, 2, 0).cpu().numpy()
    img2 = img2.permute(1, 2, 0).cpu().numpy()
    return ssim(img1, img2, multichannel=True)

def compute_psnr(img1, img2):
    img1 = img1.permute(1, 2, 0).cpu().numpy()
    img2 = img2.permute(1, 2, 0).cpu().numpy()
    return psnr(img1, img2)

def compute_fid(real_images, fake_images):
    transform = Compose([Resize((299, 299)), ToTensor(), Normalize([0.5]*3, [0.5]*3)])
    model = inception_v3(pretrained=True, transform_input=False).eval().cuda()
    def get_activations(images):
        acts = []
        for img in images:
            img = transform(img).unsqueeze(0).cuda()
            with torch.no_grad():
                act = model(img)[0].cpu().numpy()
            acts.append(act)
        return np.array(acts)

    act1 = get_activations(real_images)
    act2 = get_activations(fake_images)

    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    ssdiff = np.sum((mu1 - mu2)**2)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean): covmean = covmean.real

    return ssdiff + np.trace(sigma1 + sigma2 - 2*covmean)
