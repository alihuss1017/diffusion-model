import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import dataset
import model

builder = dataset.BuildLoaders('data', 0.7, 0.3, 64, 30000)
train_loader, val_loader = builder.build()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_time_steps = 50
batch_size = 64

def compute_alphas():
    t = torch.arange(0, max_time_steps, 1)
    s = 0.008
    f_t = torch.cos((((t / t[-1]) + s) / (1 + s)) * (torch.pi / 2)) ** 2 # (50, )
    alpha_t = f_t / f_t[0] # (50, )
    return alpha_t.cumprod(dim = 0) # (50, )

def sample_alphas(alpha_t):
    indices = torch.randint(0, max_time_steps, (batch_size, )) # (batch_size, )
    alphas = alpha_t[indices] # (batch size, )
    alphas = alphas.reshape(batch_size, 1, 1, 1) # (batch_size, 1, 1, 1)
    return indices, alphas

class Trainer:
    def __init__(self, num_epochs: int, lr: float, unet: model.UNet, device: str):
        self.num_epochs = num_epochs
        self.lr = lr
        self.unet = unet.to(device)
        self.device = device
        self.alpha_t = compute_alphas().to(device)
        self.optimizer = optim.Adam(self.unet.parameters(), lr = self.lr)
        self.loss_fn = nn.MSELoss()

    def train(self):
        self.unet.train()
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            for X in train_loader:
                self.optimizer.zero_grad()

                X = X.to(self.device)
                normalized_X = (X - 127.5) / 127.5

                indices, alphas = sample_alphas(self.alpha_t)

                eps = torch.normal(0, 1, size = normalized_X.shape).to(device)
                noisy_images = torch.sqrt(alphas) * normalized_X + torch.sqrt(1 - alphas) * eps
                noise = self.unet(noisy_images, timesteps = indices)

                loss = self.loss_fn(noise, eps)
                epoch_loss += loss.item()
                loss.backward()

                self.optimizer.step()
            
            print(f'Epoch {epoch + 1}: Loss: {epoch_loss / len(train_loader)}')

    def save_model(self):
        torch.save(self.unet.state_dict(), 'model.pt')

