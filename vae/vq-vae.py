"""
https://arxiv.org/abs/1609.02200
Discrete Variational Autoencoders

https://arxiv.org/abs/1711.00937
Neural Discrete Representation Learning

"""


import torch
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class VQ_VAE(nn.Module):
    def __init__(self, input_channels, emb_dim, n_emb, beta):
        super(VQ_VAE, self).__init__()
        self.n_emb = n_emb
        self.emb_dim = emb_dim
        self.beta = beta
        self.latent_shape = [1, 64, 8, 8]
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, emb_dim, kernel_size=4, stride=2, padding=1),
        )
        
        self.encoder_residual_stack = nn.Sequential(
            ResidualBlock(emb_dim, emb_dim),
            ResidualBlock(emb_dim, emb_dim),
        )
        
        # Decoder network
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(emb_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=8, stride=2, padding=1),
        )

        self.decoder_residual_stack = nn.Sequential(
            ResidualBlock(emb_dim, emb_dim),
            ResidualBlock(emb_dim, emb_dim),
        )

        self.embedding = nn.Embedding(n_emb, emb_dim)

    def encode(self, x):
        z = self.encoder_residual_stack(self.encoder(x))
        return z
    
    def decode(self, z):
        return self.decoder(self.decoder_residual_stack(z))

    def quantize(self, z):
        self.latent_shape = z.shape
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.emb_dim)

        d = torch.cdist(z_flattened, self.embedding.weight, p=2)

        ind = torch.argmin(d, dim=1)
        z_q = torch.index_select(self.embedding.weight, 0, ind).view(z.shape)

        e_latent_loss = F.mse_loss(z_q.detach(), z)
        q_latent_loss = F.mse_loss(z_q, z.detach())
        embed_loss = q_latent_loss + self.beta * e_latent_loss
        z_q = z + (z_q - z).detach()

        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, embed_loss

    def forward(self, x):
        z = self.encode(x)
        z_q, embed_loss = self.quantize(z)
        out = self.decode(z_q)
        return out, z_q, embed_loss
    
    @torch.no_grad
    def sample(self, num_samples, device):
        latent_shape = list(self.latent_shape)  # Convert torch.Size to a list
        latent_shape[0] = num_samples  # Modify the first element
        latent_shape = torch.Size(latent_shape)  # Convert back to torch.Size if needed

        # Step 2: Generate a latent vector
        sampled_latent = torch.randn(latent_shape).to(device)

        # Step 3: Quantize the latent vector
        quantized_latent, _ = self.quantize(sampled_latent)

        # Step 4: Decode the quantized latent vector
        generated_sample = self.decode(quantized_latent)

        return generated_sample
    

class FlattenTransform:
    def __call__(self, sample):
        return sample.view(-1)

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    # FlattenTransform()
])

# Device configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Hyperparameters
emb_dim = 64
n_emb = 256
batch_size = 64
num_epochs = 8
learning_rate = 3e-4
beta = 0.25


# MNIST dataset
mnist_trainset = MNIST(root='./data', train=True, download=True, transform=transform)
mnist_testset = MNIST(root='./data', train=False, download=True, transform=transform)

# Data loader
train_loader = DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)

# Model initialization
img_size = mnist_trainset.data.shape[-1] ** 2
vae = VQ_VAE(1, emb_dim, n_emb, beta).to(device)

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
    for i, (samples, _) in enumerate(train_loader):
        samples = samples.to(device)
        
        optimizer.zero_grad()
        
        out, z_q, embed_loss = vae(samples)
        recon_loss = F.mse_loss(out, samples)
        loss = recon_loss + embed_loss
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        losses.append(loss.item())

        # print(f"{i} {epoch_loss / (i+1)}")

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
