import open_clip
import torch.nn as nn
import torch

class OpenCLIPTextEncoder(nn.Module):
    def __init__(self, model_name="ViT-bigG-14", pretrained="laion2b_s39b_b160k", device='cuda'):
        super().__init__()
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained,
            device=device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.device = device

    def forward(self, texts):
        tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
        return text_features  # shape: [batch_size, 1024]
