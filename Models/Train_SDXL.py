import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm

from vae import AutoencoderKL
from unet import UNetModel
from text_encoder import OpenCLIPTextEncoder
from scheduler import DDPMScheduler

# --- Hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
image_size = 256
batch_size = 4
lr = 1e-4
num_epochs = 50

# --- Models
vae = AutoencoderKL().to(device)
unet = UNetModel().to(device)
text_encoder = OpenCLIPTextEncoder().to(device)
scheduler = DDPMScheduler()

# --- Optimizer & Loss
optimizer = optim.Adam(unet.parameters(), lr=lr)
mse_loss = nn.MSELoss()

# --- Dataset (use your own disaster dataset here)
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
dataset = datasets.ImageFolder("your_dataset_path", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- Training loop
for epoch in range(num_epochs):
    unet.train()
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    
    for batch in pbar:
        images, _ = batch
        images = images.to(device)

        # Step 1: Encode image into latent space
        with torch.no_grad():
            latents = vae.encode(images).sample() * 0.18215  # scaling

        # Step 2: Sample timestep and add noise
        noise = torch.randn_like(latents)
        timesteps = scheduler.get_timesteps(batch_size).to(device)
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)

        # Step 3: Get text embeddings
        text_prompts = ["a flooded village"] * batch_size  # example prompt
        text_embeds = text_encoder(text_prompts)

        # Step 4: Predict noise using U-Net
        noise_pred = unet(noisy_latents, timesteps, context=text_embeds)

        # Step 5: Compute loss and update weights
        loss = mse_loss(noise_pred, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=loss.item())

# --- Save model
torch.save(unet.state_dict(), "unet_sdxl.pth")
