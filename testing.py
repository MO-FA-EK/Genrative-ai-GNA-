import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import matplotlib.pyplot as plt

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
nz = 100
nc = 3
image_size = 32

# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1)

def load_model(model_class, path):
    model = model_class().to(device)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def process_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img = Image.open(image_path).convert('RGB')
    img_t = transform(img).unsqueeze(0).to(device)
    return img_t

def unnormalize(tensor):
    img = tensor.detach().cpu().squeeze(0)
    img = (img + 1) / 2  # [-1, 1] -> [0, 1]
    return img

def plot_images(tensors, titles, figsize=(10, 5)):
    plt.figure(figsize=figsize)
    for i, (tensor, title) in enumerate(zip(tensors, titles)):
        img = unnormalize(tensor)
        img = transforms.ToPILImage()(img)
        plt.subplot(1, len(tensors), i + 1)
        plt.imshow(img)
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

def main():
    # Paths
    generator_path = 'gan_output/generator.pth'
    discriminator_path = 'gan_output/discriminator.pth'

    # Load models
    generator = load_model(Generator, generator_path)
    discriminator = load_model(Discriminator, discriminator_path)

    # Generate fake image
    noise = torch.randn(1, nz, 1, 1, device=device)
    fake_image = generator(noise)
    torch.save(fake_image, 'generated_tensor.pt')
    print("✅ Fake image generated.")

    # Discriminator on fake image
    output_fake = discriminator(fake_image)
    print(f"Discriminator output for generated image: {output_fake.item():.4f} → {'REAL' if output_fake.item() > 0.5 else 'FAKE'}")

    # Try a real user image
    user_image_path = 'cat.jpg'  # Replace this if needed
    if os.path.exists(user_image_path):
        real_image = process_image(user_image_path)
        output_real = discriminator(real_image)
        print(f"Discriminator output for your image: {output_real.item():.4f} → {'REAL' if output_real.item() > 0.5 else 'FAKE'}")

        # Plot both images
        plot_images([fake_image, real_image], ["Generated (Fake)", "Your Image (Real)"])
    else:
        # Only show generated image
        plot_images([fake_image], ["Generated (Fake)"])
        print(f"⚠️ Custom image not found at path: {user_image_path}")

if __name__ == "__main__":
    main()
