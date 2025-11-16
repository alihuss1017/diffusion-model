import torch
import torch.nn as nn
import torch.optim as optim
import model


def compute_alphas(num_time_steps: int):
    t = torch.arange(0, num_time_steps, 1)
    s = 0.008
    f_t = torch.cos((((t / t[-1]) + s) / (1 + s)) * (torch.pi / 2)) ** 2 # (50, )
    alpha_t = f_t / f_t[0] # (50, )
    return alpha_t.cumprod(dim = 0) # (50, )

def sample_alphas(num_time_steps: int, alpha_t, batch_size: int):
    indices = torch.randint(0, num_time_steps, (batch_size, )) # (batch_size, )
    alphas = alpha_t[indices] # (batch size, )
    alphas = alphas.reshape(batch_size, 1, 1, 1) # (batch_size, 1, 1, 1)
    return indices, alphas

class Trainer:
    def __init__(self, num_epochs: int, lr: float, unet: model.UNet, device: str,
                 train_loader, val_loader, batch_size: int, num_time_steps: int):
        self.num_epochs = num_epochs
        self.lr = lr
        self.unet = unet.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.batch_size = batch_size
        self.num_time_steps = num_time_steps
        self.alpha_t = compute_alphas(num_time_steps).to(device)
        self.optimizer = optim.Adam(self.unet.parameters(), lr = self.lr)
        self.loss_fn = nn.MSELoss()

    def train(self):
        self.unet.train()
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            for X in self.train_loader:
                self.optimizer.zero_grad()

                X = X.to(self.device)
                normalized_X = (X - 127.5) / 127.5

                indices, alphas = sample_alphas(self.num_time_steps, self.alpha_t, self.batch_size)

                eps = torch.normal(0, 1, size = normalized_X.shape).to(self.device)
                noisy_images = torch.sqrt(alphas) * normalized_X + torch.sqrt(1 - alphas) * eps
                noise = self.unet(noisy_images, timesteps = indices)

                loss = self.loss_fn(noise, eps)
                epoch_loss += loss.item()
                loss.backward()

                self.optimizer.step()
            
            print(f'Epoch {epoch + 1}: Loss: {epoch_loss / len(self.train_loader)}')

    def save_model(self):
        torch.save(self.unet.state_dict(), 'model.pt')

