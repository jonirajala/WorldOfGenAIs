"""
https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

based on the gpt-2 architecture
"""

import torch
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import math
from tqdm.auto import tqdm
from torchtune.modules import RMSNorm, RotaryPositionalEmbeddings


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        scaled_hidden = int(2 / 3 * 4 * config.emb_dim)
        self.fc1 = nn.Linear(config.emb_dim, scaled_hidden, bias=False)
        self.fc2 = nn.Linear(config.emb_dim, scaled_hidden, bias=False)
        self.fc3 = nn.Linear(scaled_hidden, config.emb_dim, bias=False)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        hidden = F.silu(x1)
        hidden = hidden * x2
        return self.fc3(hidden)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.emb_dim % config.n_head == 0
        self.emb_dim = config.emb_dim
        self.n_head = config.n_head
        self.head_dim = config.emb_dim // config.n_head
        self.batch_size = config.batch_size
        self.block_size = config.block_size

        self.Wq = nn.Linear(config.emb_dim, self.n_head * self.head_dim, bias=False)
        self.Wk = nn.Linear(config.emb_dim, self.n_head * self.head_dim, bias=False)
        self.Wv = nn.Linear(config.emb_dim, self.n_head * self.head_dim, bias=False)
        self.Wo = nn.Linear(config.emb_dim, self.n_head * self.head_dim, bias=False)
        self.pos_emb = RotaryPositionalEmbeddings(self.head_dim, config.block_size)

        self.cache_k = None
        self.cache_v = None


        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
        

    def forward(self, x, start_pos):
        batch_size, seq_len, dim = x.shape
        assert dim == self.emb_dim, "dim must be equal to self.emb_dim"
        if start_pos == 0 or self.cache_k is None or self.cache_v is None:
            self.cache_k = torch.zeros((batch_size, self.block_size, self.n_head, self.head_dim), device=x.device)
            self.cache_v = torch.zeros((batch_size, self.block_size, self.n_head, self.head_dim), device=x.device)

        xq = self.Wq(x)
        xk = self.Wk(x)
        xv = self.Wv(x)

        xq = xq.view(batch_size, seq_len, self.n_head, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_head, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_head, self.head_dim)

        xq = self.pos_emb(xq)
        xk = self.pos_emb(xk)

        self.cache_k[:batch_size, start_pos:start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos:start_pos + seq_len] = xv

        
        keys = self.cache_k[:batch_size, :start_pos + seq_len]
        values = self.cache_v[:batch_size, :start_pos + seq_len]
        
        queries = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        
         # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            context = torch.nn.functional.scaled_dot_product_attention(queries, keys, values, attn_mask=None, dropout_p=0 if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (queries @ keys.transpose(-2, -1)) * (1.0 / math.sqrt(keys.size(-1)))
            att = att.masked_fill(self.bias[:,:,:seq_len,:seq_len] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            context = att @ values # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)


        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.Wo(context)
        return output


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rn1 = RMSNorm(config.emb_dim)
        self.rn2 = RMSNorm(config.emb_dim)
        self.attn = MultiHeadSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x, start_pos):
        x = x + self.attn(self.rn1(x), start_pos)
        x = x + self.mlp(self.rn2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.inp_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.fc_out = nn.Linear(config.emb_dim, config.vocab_size, bias=False)
        self.rmsnorm = RMSNorm(config.emb_dim)
        # self.inp_emb.weight = self.fc_out.weight # https://paperswithcode.com/method/weight-tying

    def forward(self, x, start_pos, y=None):
        batch, seq_len = x.shape
        x = self.inp_emb(x)
        for block in self.blocks:
            x = block(x, start_pos)
        x = self.rmsnorm(x)

        logits = self.fc_out(x)
        loss = None
        if y is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, inp, temperature=1.0, top_k=10):
        inp = inp.reshape(1, -1)
        start_pos = 0
        for _ in range(self.config.block_size - inp.shape[1]):
            logits, _ = self.forward(inp[:, start_pos:], start_pos)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            inp_next = torch.multinomial(probs, num_samples=1)
            inp = torch.cat((inp, inp_next), dim=1)
            start_pos = inp.shape[1] - 1
            # print(inp.shape)

        return inp[0]

class Config:
    def __init__(
        self, vocab_size, emb_dim, n_layers, n_head, num_experts=None, top_k=None, d_inner=None, d_conv=None, dt_rank=None, d_state=None
    ):
        self.block_size = 28*28
        self.window_size = self.block_size // 2
        self.batch_size = 32
        self.iters = 1000
        self.dropout = 0.1
        self.n_kv_heads = 8
        self.vocab_size = vocab_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_inner = d_inner
        self.d_conv = d_conv
        self.dt_rank = dt_rank
        self.d_state = d_state

class FlattenAndQuantizeTransform:
    def __call__(self, sample):
        # Normalize to [0, 1]
        sample = transforms.functional.to_pil_image(sample).convert("L")
        sample = transforms.functional.to_tensor(sample)
        sample = sample * 255.0
        sample = sample.long().view(-1)  # Flatten and convert to integers
        return sample
    
# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    FlattenAndQuantizeTransform()
])

# Device configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Hyperparameters
batch_size = 16
num_epochs = 1
learning_rate = 3e-4

# MNIST dataset
mnist_trainset = MNIST(root='./data', train=True, download=True, transform=transform)
mnist_testset = MNIST(root='./data', train=False, download=True, transform=transform)

# Data loader
train_loader = DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)

# Model initialization
img_size = mnist_trainset.data.shape[-1] ** 2
img, label = next(iter(train_loader))
config = Config(vocab_size=255, emb_dim=256, n_head=8, n_layers=8)
gpt = GPT(config).to(device)

# Loss and optimizer
optimizer = optim.AdamW(gpt.parameters(), lr=learning_rate)

# Training the model
losses = []

for epoch in range(num_epochs):
    epoch_loss = 0
    pbar = tqdm(range(len(train_loader)), desc="Training Progress")
    for i in pbar:
        samples, _ = next(iter(train_loader))
        x, y = samples[:, :-1], samples[:, 1:]
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        out, loss = gpt(x=x, y=y, start_pos=0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gpt.parameters(), 1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        losses.append(loss.item())
        pbar.set_postfix({"train_loss": loss.item()})
        
        if i == 1875:
            break

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}")

# Plotting the loss
plt.figure(figsize=(10, 5))
plt.plot(losses, label='Loss')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.legend()
plt.title('gpt Loss During Training')
plt.show()


# Define the function to generate and display images
def generate_images(model, num_images=5, temperature=1.0, top_k=None):
    model.eval()  # Set the model to evaluation mode
    
    generated_images = []
    for i in range(num_images):
        start_token = torch.zeros((1, 1), dtype=torch.long).to(device)  # Starting token
        generated_sequence = model.generate(start_token, temperature=temperature, top_k=top_k)
        generated_images.append(generated_sequence.cpu().numpy())
        print(f"{i} images generated")
    
    return generated_images

# Generate new images
num_images = 2
generated_images = generate_images(gpt, num_images=num_images, temperature=1.0, top_k=50)

# Reshape and plot the generated images
fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
for i, img in enumerate(generated_images):
    img = img.reshape(28, 28)  # Reshape to 28x28 image
    axes[i].imshow(img, cmap='gray')
    axes[i].axis('off')

plt.show()