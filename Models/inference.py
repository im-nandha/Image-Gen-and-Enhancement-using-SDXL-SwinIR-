import torch
from PIL import Image
from torchvision import transforms

# Import your models (assumed implemented in separate files)
from models.stable_diffusion_xl import StableDiffusionXL
from models.swinir import SwinIR

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load pre-trained models (or your checkpoints)
sdxl_model = StableDiffusionXL().to(device)
sdxl_model.load_state_dict(torch.load("sdxl.pth"))
sdxl_model.eval()

swinir_model = SwinIR().to(device)
swinir_model.load_state_dict(torch.load("swinir.pth"))
swinir_model.eval()

# Text prompt for generation
prompt = "A serene mountain landscape during sunrise"

# Step 1: Generate image with Stable Diffusion XL
with torch.no_grad():
    generated_image = sdxl_model.generate(prompt)  # Implement generate method in SDXL model

# Assume generated_image is a tensor [C,H,W] normalized [0,1]
# Convert to batch and send to device for SwinIR
input_for_swinir = generated_image.unsqueeze(0).to(device)

# Step 2: Enhance generated image using SwinIR
with torch.no_grad():
    enhanced_image = swinir_model(input_for_swinir).squeeze(0).clamp(0, 1)

# Convert tensor to PIL Image to save or display
to_pil = transforms.ToPILImage()
final_img = to_pil(enhanced_image.cpu())
final_img.save("final_output.png")

print("Generated image enhanced and saved as final_output.png")
