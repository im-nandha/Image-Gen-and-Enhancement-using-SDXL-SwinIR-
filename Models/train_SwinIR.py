import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from models.swinir import SwinIR
from utils.metrics import compute_psnr, compute_ssim
from torchvision.datasets import ImageFolder

device = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset & DataLoader
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = ImageFolder(root="path_to_train_images", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Model
model = SwinIR().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()

for epoch in range(20):
    model.train()
    epoch_loss = 0
    for inputs, _ in train_loader:
        inputs = inputs.to(device)
        # Assume inputs are noisy; targets are clean images (for demo)
        targets = inputs.clone()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(train_loader)}")

torch.save(model.state_dict(), "swinir.pth")
