import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import multiprocessing

# Hyperparameters
batch_size = 100
image_size = 784
z_dim = 100
hidden_dim = 256
num_epochs = 20
lr = 0.0002

# Generator
class Generator(nn.Module):
    def __init__(self, z_dim, hidden_dim, image_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, image_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, image_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# Loss
criterion = nn.BCELoss()

# Transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Dataset and Dataloader
dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# Initialize models
G = Generator(z_dim, hidden_dim, image_size)
D = Discriminator(image_size, hidden_dim)

# Optimizers
optimizer_G = optim.Adam(G.parameters(), lr=lr)
optimizer_D = optim.Adam(D.parameters(), lr=lr)

# Move models to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
G = G.to(device)
D = D.to(device)

# Training Loop
def train():
    g_losses = []
    d_losses = []

    for epoch in range(num_epochs):
        for i, (real_images, _) in enumerate(dataloader):
            real_images = real_images.view(-1, image_size).to(device)
            batch_size = real_images.size(0)

            # Labels
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # Train Discriminator
            outputs = D(real_images)
            d_loss_real = criterion(outputs, real_labels)

            z = torch.randn(batch_size, z_dim).to(device)
            fake_images = G(z)
            outputs = D(fake_images.detach())
            d_loss_fake = criterion(outputs, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            z = torch.randn(batch_size, z_dim).to(device)
            fake_images = G(z)
            outputs = D(fake_images)
            g_loss = criterion(outputs, real_labels)

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

            # Save losses
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

            # Logging
            if (i+1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], "
                      f"Loss D: {d_loss.item():.4f}, Loss G: {g_loss.item():.4f}")

    # Save and plot loss curves
    torch.save({'g_losses': g_losses, 'd_losses': d_losses}, 'loss_history.pt')
    plot_losses(g_losses, d_losses)

def plot_losses(g_losses, d_losses):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_losses, label="Generator Loss")
    plt.plot(d_losses, label="Discriminator Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_plot.png")
    plt.show()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    train()
