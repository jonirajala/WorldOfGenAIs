"""
https://github.com/huggingface/diffusion-models-class
https://arxiv.org/abs/1503.03585 og diffusion
https://arxiv.org/abs/2006.11239 ddpm
https://arxiv.org/abs/2102.09672 improved ddpm
"""

import torch
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision


class BasicUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.down_layers = torch.nn.ModuleList([ 
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
        ])
        self.up_layers = torch.nn.ModuleList([
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, out_channels, kernel_size=5, padding=2), 
        ])
        self.act = nn.SiLU() # The activation function
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2)
    
    def forward(self, x):
        h = []
        for i, l in enumerate(self.down_layers):
            x = self.act(l(x))
            if i < 2:
              h.append(x)
              x = self.downscale(x)
              
        for i, l in enumerate(self.up_layers):
            if i > 0:
              x = self.upscale(x)
              x += h.pop()
            x = self.act(l(x))
            
        return x


class FlattenTransform:
    def __call__(self, sample):
        return sample.view(-1)

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,)),
    # FlattenTransform()
])


def corrupt(img, amount):
    noise = torch.rand_like(img)
    amount = amount.view(-1, 1, 1, 1)
    return img*(1-amount) + noise*amount 

# Device configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Hyperparameters
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
unet = BasicUNet().to(device)


# x, y = next(iter(train_loader))

# fig, axs = plt.subplots(2, 1, figsize=(12, 5))
# axs[0].set_title('Input data')
# axs[0].imshow(torchvision.utils.make_grid(x)[0], cmap='Greys')

# # Adding noise
# amount = torch.linspace(0, 1, x.shape[0]) # Left to right -> more corruption
# noised_x = corrupt(x, amount)

# # Plotting the noised version
# axs[1].set_title('Corrupted data (-- amount increases -->)')
# axs[1].imshow(torchvision.utils.make_grid(noised_x)[0], cmap='Greys');


# plt.show()


# # Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(unet.parameters(), lr=learning_rate)

# Training the model
losses = []

for epoch in range(num_epochs):
    epoch_loss = 0
    for samples, _ in train_loader:
        samples = samples.to(device)
        noise_amount = torch.rand(samples.shape[0]).to(device) 
        noisy_samples = corrupt(samples, noise_amount) 

        optimizer.zero_grad()
        
        out = unet(noisy_samples)
        loss = criterion(out, samples)

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

# Fetch some data
x, y = next(iter(train_loader))
x = x[:8] # Only using the first 8 for easy plotting

# Corrupt with a range of amounts
amount = torch.linspace(0, 1, x.shape[0]) # Left to right -> more corruption
noised_x = corrupt(x, amount)

# Get the model predictions
with torch.no_grad():
  preds = unet(noised_x.to(device)).detach().cpu()

# Plot
fig, axs = plt.subplots(3, 1, figsize=(12, 7))
axs[0].set_title('Input data')
axs[0].imshow(torchvision.utils.make_grid(x)[0].clip(0, 1), cmap='Greys')
axs[1].set_title('Corrupted data')
axs[1].imshow(torchvision.utils.make_grid(noised_x)[0].clip(0, 1), cmap='Greys')
axs[2].set_title('Network Predictions')
axs[2].imshow(torchvision.utils.make_grid(preds)[0].clip(0, 1), cmap='Greys');
plt.show()