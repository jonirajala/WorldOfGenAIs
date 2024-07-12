"""
https://arxiv.org/pdf/1406.2661
Generative Adversarial Nets
simultaneously train two models: a generative model G
that captures the data distribution, and a discriminative model D that estimates
the probability that a sample came from the training data rather than G

. In the case where G and D are defined
by multilayer perceptrons, the entire system can be trained with backpropagation

"""


import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


class Generator(nn.Module):
    def __init__(self, fan_in, fan_out):
        super().__init__()
        self.fc1 = nn.Linear(fan_in, 32)
        self.fc2 = nn.Linear(32, fan_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        y = self.fc2(x)

        return y

class Discriminator(nn.Module):
    def __init__(self, fan_in):
        super().__init__()
        self.fc1 = nn.Linear(fan_in, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        y = F.sigmoid(x)
        return y
    

device = "mps"


noise_dist = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
targ_dist = torch.distributions.chi2.Chi2(torch.tensor([10.0]), validate_args=None)

generator = Generator(1, 1).to(device)
discriminator = Discriminator(1).to(device)
criterion = nn.BCELoss()
d_optim = optim.Adam(discriminator.parameters(), lr=3e-4)
g_optim = optim.Adam(generator.parameters(), lr=3e-4)

batch_size = 64
num_epochs = 10000

# Lists to store losses
d_losses = []
g_losses = []

for i in range(num_epochs):
    real_samples, real_targ = targ_dist.sample((batch_size//2,)).to(device), torch.ones((batch_size//2,1)).to(device)
    noise_sample, fake_targ = noise_dist.sample((batch_size//2,)).to(device), torch.zeros((batch_size//2,1)).to(device)

    if i % 2 == 0:
        # discriminator
        fake_samples = generator(noise_sample)
        d_optim.zero_grad()
        real_out = discriminator(real_samples)
        fake_out = discriminator(fake_samples.detach())
        real_loss = criterion( real_out, real_targ)
        fake_loss = criterion(fake_out, fake_targ)
        d_loss = real_loss + fake_loss
        d_loss.backward()
        d_optim.step()

    # for j in range(3):
    # generator
    g_optim.zero_grad()
    fake_samples = generator(noise_sample)
    fake_out = discriminator(fake_samples)
    g_loss = criterion(fake_out, real_targ)
    g_loss.backward()
    g_optim.step()

    d_losses.append(d_loss.item())
    g_losses.append(g_loss.item())

    if i % 100 == 0:
        print(f"{i} d_loss: {d_loss:.4f}, g_loss: {g_loss:.4f}")


plt.figure(figsize=(10, 5))
plt.plot(d_losses, label='Discriminator Loss')
plt.plot(g_losses, label='Generator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('GAN Losses During Training')
plt.show()


noise_sample = noise_dist.sample((1000,)).to(device)
fake_samples = generator(noise_sample)
real_samples = targ_dist.sample((1000,)).to(device)

plt.figure(figsize=(10, 5))
plt.hist(fake_samples.cpu().detach().numpy(), bins=50, alpha=0.5, label='Generated chi2')
plt.hist(real_samples.cpu().detach().numpy(), bins=50, alpha=0.5, label='Real chi2')
plt.legend()
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Distribution of Real and Generated Samples')
plt.show()