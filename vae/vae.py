"""
https://arxiv.org/pdf/1312.6114
Auto-Encoding Variational Bayes


"""

import torch
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 2),
        )
        self.fc_mu = nn.Linear(input_dim // 2, latent_dim)
        self.fc_var = nn.Linear(input_dim // 2, latent_dim)
        
        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, input_dim),
        )

    def encode(self, x):
        x = self.encoder(x)
        mu, var = self.fc_mu(x), self.fc_var(x)
        return mu, var

    def reparameterize(self, mu, var):
        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, var = self.encode(x)
        z = self.reparameterize(mu, var)
        out = self.decode(z)
        return out, mu, var

    def sample(self, num_samples, device):
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples

class FlattenTransform:
    def __call__(self, sample):
        return sample.view(-1)

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    FlattenTransform()
])

# Device configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Hyperparameters
latent_dim = 50
batch_size = 64
num_epochs = 5
learning_rate = 3e-4

# MNIST dataset
mnist_trainset = MNIST(root='./data', train=True, download=True, transform=transform)
mnist_testset = MNIST(root='./data', train=False, download=True, transform=transform)

# Data loader
train_loader = DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)

# Model initialization
img_size = mnist_trainset.data.shape[-1] ** 2
vae = VAE(img_size, latent_dim).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

# KL Divergence calculation
def kl_divergence(mu, var):
    return -0.5 * torch.sum(1 + var - mu.pow(2) - var.exp())

# Training the model
losses = []

for epoch in range(num_epochs):
    epoch_loss = 0
    for samples, _ in train_loader:
        samples = samples.to(device)
        
        optimizer.zero_grad()
        
        out, mu, var = vae(samples)
        recon_loss = criterion(out, samples)
        kl_loss = kl_divergence(mu, var)
        loss = recon_loss + kl_loss
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        losses.append(loss.item())
        
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}")

# Plotting the loss
plt.figure(figsize=(10, 5))
plt.plot(losses, label='Loss')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.legend()
plt.title('VAE Loss During Training')
plt.show()

# Sampling and plotting images from the VAE
num_samples = 8
samples = vae.sample(num_samples, device).cpu().detach().numpy()

fig, axes = plt.subplots(1, num_samples, figsize=(num_samples, 1))
for i, ax in enumerate(axes):
    ax.imshow(samples[i].reshape(28, 28), cmap='gray')
    ax.axis('off')
plt.suptitle('Sampled Images from VAE')
plt.show()
