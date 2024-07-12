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
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader


class Generator(nn.Module):
    def __init__(self, fan_in, fan_out):
        super().__init__()
        self.fc1 = nn.Linear(fan_in, fan_out//2)
        self.fc2 = nn.Linear(fan_out//2, fan_out//2)
        self.fc3 = nn.Linear(fan_out//2, fan_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class Discriminator(nn.Module):
    def __init__(self, fan_in):
        super().__init__()
        self.fc1 = nn.Linear(fan_in, fan_in//2)
        self.fc2 = nn.Linear(fan_in//2, fan_in//2)
        self.fc3 = nn.Linear(fan_in//2, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        x = F.sigmoid(x)
        return x
    
class FlattenTransform:
    def __call__(self, sample):
        return sample.view(-1)  # Flatten the image
# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    FlattenTransform()  # Custom transformation to flatten the image

])


device = "mps"
noise_dim = 100
batch_size = 64
num_epochs = 5

mnist_trainset = MNIST(root='./data', train=True, download=True, transform=transform)
mnist_testset = MNIST(root='./data', train=False, download=True, transform=None)
# mnist_trainset, mnist_testset = mnist_trainset.flatten(-2), mnist_testset.flatten(-2)
# print(mnist_testset.shape)
img_size = mnist_trainset.data.shape[-1] ** 2

train_loader = DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)
print(img_size)

generator = Generator(noise_dim,img_size).to(device)
discriminator = Discriminator(img_size).to(device)
criterion = nn.BCELoss()
d_optim = optim.Adam(discriminator.parameters(), lr=3e-4)
g_optim = optim.Adam(generator.parameters(), lr=3e-4)



# Lists to store losses
d_losses = []
g_losses = []

for i in range(num_epochs):
    for real_samples, _ in train_loader:
        if len(real_samples) != batch_size:
            continue
        real_samples = real_samples.to(device)
        real_targ = torch.ones((batch_size,1)).to(device)
        noise_sample, fake_targ = torch.randn(batch_size, noise_dim).to(device), torch.zeros((batch_size,1)).to(device)
        
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

    print(i)
    print(f"{i} d_loss: {d_loss:.4f}, g_loss: {g_loss:.4f}")
        


plt.figure(figsize=(10, 5))
plt.plot(d_losses, label='Discriminator Loss')
plt.plot(g_losses, label='Generator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('GAN Losses During Training')
plt.show()


def generate_and_plot_images(generator, num_images, noise_dim):
    generator.eval()  # Set generator to evaluation mode
    noise = torch.randn(num_images, noise_dim).to(device)
    generated_images = generator(noise).cpu().detach().numpy()

    # Reshape to 28x28 images
    generated_images = generated_images.reshape(num_images, 28, 28)

    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(generated_images[i], cmap='gray')
        plt.axis('off')
    plt.show()

# Generate and plot 5 images
generate_and_plot_images(generator, 5, noise_dim)