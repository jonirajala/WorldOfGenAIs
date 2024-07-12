"""
https://arxiv.org/pdf/1511.06434

UNSUPERVISED REPRESENTATION LEARNING
WITH DEEP CONVOLUTIONAL
GENERATIVE ADVERSARIAL NETWORKS



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
    def __init__(self, fan_in, output_channels=1):
        super().__init__()
        self.fc1 = nn.Linear(fan_in, 256 * 7 * 7)
        self.conv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)  # Output: (128, 14, 14)
        self.conv2 = nn.ConvTranspose2d(128, output_channels, 4, 2, 1, bias=False)  # Output: (output_channels, 28, 28)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x).view(-1, 256, 7, 7)
        x = self.relu(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.tanh(self.conv2(x))
        return x
class Discriminator(nn.Module):
    def __init__(self, input_channels=1):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, 4, 2, 1, bias=False)
        self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.lrelu2 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.lrelu3 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv4 = nn.Conv2d(256, 512, 3, 1, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.lrelu4 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv5 = nn.Conv2d(512, 1, 3, 1, 0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.lrelu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lrelu2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.lrelu3(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.lrelu4(x)

        x = self.conv5(x)
        x = self.sigmoid(x)
        x = x.view(x.size(0), -1)
        
        return x
    

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),

])


device = "mps"
noise_dim = 100
batch_size = 64
num_epochs = 3

mnist_trainset = MNIST(root='./data', train=True, download=True, transform=transform)
mnist_testset = MNIST(root='./data', train=False, download=True, transform=None)
# mnist_trainset, mnist_testset = mnist_trainset.flatten(-2), mnist_testset.flatten(-2)
# print(mnist_testset.shape)
img_size = mnist_trainset.data.shape[-1] ** 2

train_loader = DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)
print(img_size)

generator = Generator(noise_dim).to(device)
discriminator = Discriminator().to(device)
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
        
        if i % 3 == 0:
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