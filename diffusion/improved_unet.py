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


class ResnetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super(ResnetBlock2D, self).__init__()
        self.inc = in_channels
        self.outc = out_channels
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(0.0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.nonlinearity = nn.SiLU()
        self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1) if in_channels != out_channels else None

    def forward(self, x, t):
        h = self.norm1(x)
        h = self.nonlinearity(h)
        h = self.conv1(h)
        h += self.time_emb_proj(t)[None, :, None, None]
        h = self.norm2(h)
        h = self.nonlinearity(h)
        h = self.conv2(self.dropout(h))
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return x + h

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        self.group_norm = nn.GroupNorm(32, channels)
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.proj_attn = nn.Linear(channels, channels)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Group normalization and reshape
        x_norm = self.group_norm(x).view(batch_size, channels, -1).permute(0, 2, 1)
        
        # Linear projections
        q = self.query(x_norm)
        k = self.key(x_norm)
        v = self.value(x_norm)
        
        # Attention weights
        attn_weights = torch.bmm(q, k.permute(0, 2, 1)) * (channels ** -0.5)
        attn_weights = attn_weights.softmax(dim=-1)
        
        # Attention output
        attn_output = torch.bmm(attn_weights, v)
        
        # Reshape back to the original shape
        attn_output = attn_output.permute(0, 2, 1).view(batch_size, channels, height, width)
        
        # Project the attention output and add to the input
        proj_output = self.proj_attn(attn_output.view(batch_size, channels, -1).permute(0, 2, 1))
        proj_output = proj_output.permute(0, 2, 1).view(batch_size, channels, height, width)
        
        return x + proj_output
class DownBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, attention=False):
        super(DownBlock2D, self).__init__()
        self.resnets = nn.ModuleList([ResnetBlock2D(in_channels, in_channels, time_emb_dim) for _ in range(2)])
        self.attentions = nn.ModuleList([AttentionBlock(in_channels) for _ in range(2)]) if attention else None
        self.downsamplers = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)])

    def forward(self, x, t):
        for resnet in self.resnets:
            x = resnet(x, t)
        if self.attentions:
            for attention in self.attentions:
                x = attention(x)
        for downsampler in self.downsamplers:
            x = downsampler(x)
        return x

class UpBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, attention=False):
        super(UpBlock2D, self).__init__()
        self.resnets = nn.ModuleList([ResnetBlock2D(in_channels, in_channels, time_emb_dim) for _ in range(3)])
        self.attentions = nn.ModuleList([AttentionBlock(in_channels) for _ in range(3)]) if attention else None
        self.upsamplers = nn.ModuleList([nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)])

    def forward(self, x, t):
        for resnet in self.resnets:
            x = resnet(x, t)
        if self.attentions:
            for attention in self.attentions:
                x = attention(x)
        for upsampler in self.upsamplers:
            x = upsampler(x)
        return x

class UNetMidBlock2D(nn.Module):
    def __init__(self, channels, time_emb_dim):
        super(UNetMidBlock2D, self).__init__()
        self.attentions = nn.ModuleList([AttentionBlock(channels)])
        self.resnets = nn.ModuleList([ResnetBlock2D(channels, channels, time_emb_dim) for _ in range(2)])

    def forward(self, x, t):
        for resnet in self.resnets:
            x = resnet(x, t)
        for attention in self.attentions:
            x = attention(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size: int, n: int = 10000, max_time_steps: int = 1000) -> None:
        super().__init__()

        i = torch.arange(embedding_size // 2)
        k = torch.arange(max_time_steps).unsqueeze(dim=1)

        self.pos_embeddings = torch.zeros(max_time_steps, embedding_size, requires_grad=False)
        self.pos_embeddings[:, 0::2] = torch.sin(k / (n ** (2 * i / embedding_size)))
        self.pos_embeddings[:, 1::2] = torch.cos(k / (n ** (2 * i / embedding_size)))

    def forward(self, t):
        return self.pos_embeddings[t, :].to(device)

class TimestepEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TimestepEmbedding, self).__init__()
        self.linear_1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, t):
        t = self.act(self.linear_1(t))
        return self.linear_2(t)

class BasicUNet(nn.Module):
    def __init__(self, sample_size=28,    # the target image resolution
                in_channels=1,            # the number of input channels, 3 for RGB images
                out_channels=1,           # the number of output channels
                layers_per_block=2,       # how many ResNet layers to use per UNet block
                block_out_channels=(32, 64, 64), # Roughly matching our basic unet example
                down_block_types=( 
                    "DownBlock2D",        # a regular ResNet downsampling block
                    "AttnDownBlock2D",    # a ResNet downsampling block with spatial self-attention
                    "AttnDownBlock2D",
                ), 
                up_block_types=(
                    "AttnUpBlock2D", 
                    "AttnUpBlock2D",      # a ResNet upsampling block with spatial self-attention
                    "UpBlock2D",          # a regular ResNet upsampling block
                )):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.time_proj = PositionalEncoding(embedding_size=32)
        self.time_embedding = TimestepEmbedding(32, 128, 128)
        self.down_blocks = nn.ModuleList([
            DownBlock2D(32, 32, 128),
            DownBlock2D(32, 64, 128, attention=True),
            DownBlock2D(64, 64, 128, attention=True)
        ])
        self.mid_block = UNetMidBlock2D(64, 128)
        self.up_blocks = nn.ModuleList([
            UpBlock2D(64, 64, 128, attention=True),
            UpBlock2D(64, 64, 128, attention=True),
            UpBlock2D(64, 32, 128)
        ])
        self.conv_norm_out = nn.GroupNorm(8, 32)  # Adjusted number of groups
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(32, out_channels, kernel_size=7, stride=1, padding=1)

    def forward(self, x, t):
        t_emb = self.time_embedding(self.time_proj(t))
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x, t_emb)
        x = self.mid_block(x, t_emb)
        for block in self.up_blocks:
            x = block(x, t_emb)
        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)
        return x
def add_noise(image, beta, timesteps):
    """
    Add noise to an image according to the DDPM process.

    Parameters:
    - image: torch.Tensor, the original image (x_0)
    - beta: torch.Tensor, an array of noise coefficients of length T
    - timesteps: int, the number of time steps

    Returns:
    - noisy_images: list of torch.Tensor, the noisy images at each time step
    """
    noisy_images = []
    noise_list = []
    for t in range(1, timesteps + 1):
        beta_t = beta[t - 1]
        noise = torch.randn_like(image)
        mean = torch.sqrt(1 - beta_t) * image
        variance = beta_t
        x_t = mean + torch.sqrt(variance) * noise
        noisy_images.append(x_t)
        noise_list.append(noise)
    
    return torch.stack(noisy_images, dim=1), torch.stack(noise_list, dim=1)


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
batch_size = 8
num_epochs = 1
learning_rate = 3e-4
timesteps = 8

# MNIST dataset
mnist_trainset = MNIST(root='./data', train=True, download=True, transform=transform)
mnist_testset = MNIST(root='./data', train=False, download=True, transform=transform)

# Data loader
train_loader = DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)

# Model initialization
img_size = mnist_trainset.data.shape[-1] ** 2
unet = BasicUNet().to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(unet.parameters(), lr=learning_rate)

# Beta schedule
beta = torch.linspace(0.1, 0.2, timesteps).to(device)

# Training the model
losses = []

for epoch in range(num_epochs):
    epoch_loss = 0
    for i, (samples, _) in enumerate(train_loader):
        samples = samples.to(device)
        
        optimizer.zero_grad()
        
        noisy_samples, noise = add_noise(samples, beta, timesteps)
        
        # Reshape noisy samples and noise to have correct dimensions for the UNet
        noisy_samples = noisy_samples.view(-1, 1, 28, 28)
        noise = noise.view(-1, 1, 28, 28)
        
        out = unet(noisy_samples, timesteps)
        
        loss = criterion(out, noise)

        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        losses.append(loss.item())
        if i % 100 == 0:
            print(f"{i} / {len(train_loader)}, loss {epoch_loss / (i+1)}")
        

        if i == 200:
            break
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}")


# Plot losses
plt.figure(figsize=(12, 5))
plt.plot(losses)
# plt.ylim(0, 0.1)
plt.title('Loss over time')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()

# Define the noise schedule (simple linear schedule for demonstration)
def linear_noise_schedule(t, n_steps):
    return 1 - (t / n_steps)

# Denoising process
n_steps = 20
x = torch.rand(2, 1, 28, 28).to(device)  # Starting from random noise
denoising_images = []

for i in range(n_steps):
    with torch.no_grad():
        t = n_steps - i - 1
        noise_scale = linear_noise_schedule(t, n_steps)
        pred_noise = unet(x, t)
        x = (x - noise_scale * pred_noise) / (1 - noise_scale)

    # Store denoised images at each step
    denoising_images.append(x.detach().cpu())

# Plot denoising process
fig, axs = plt.subplots(4, 5, figsize=(20, 16))  # Adjusted the layout and size
for i, ax in enumerate(axs.flatten()):
    if i < len(denoising_images):
        img_grid = torchvision.utils.make_grid(denoising_images[i], nrow=2, padding=1, normalize=True)
        ax.imshow(img_grid.permute(1, 2, 0))
        ax.axis('off')
    else:
        ax.remove()

plt.suptitle('Denoising Process Over Time', fontsize=20)
plt.show()