import torch
import model

def compute_alphas(num_time_steps: int):
    t = torch.arange(0, num_time_steps, 1)
    s = 0.008
    f_t = torch.cos((((t / t[-1]) + s) / (1 + s)) * (torch.pi / 2)) ** 2 # (50, )
    alpha_t = f_t / f_t[0] # (50, )
    return alpha_t, alpha_t.cumprod(dim = 0) # (50, ), (50, )

class InferenceEngine:
    def __init__(self, num_time_steps: int, batch_size: int, unet: model.UNet):
        self.num_time_steps = num_time_steps
        self.batch_size = batch_size
        self.unet = unet
        self.alpha_t, self.alpha_bar_t = compute_alphas(num_time_steps)

    def run(self):

        samples = torch.normal(0, 1, size = (self.batch_size, 3, 64, 64))
        x_t = samples
        self.unet.eval()

        with torch.no_grad():
            for t in range(self.num_time_steps - 1, 0 - 1, -1):
                z = torch.normal(0, 1, size = (self.batch_size, 3, 64, 64)) if t > 0 else 0

                tensorized_t = torch.tensor([t])
                predicted_noise = self.unet(x_t, tensorized_t)

                beta_t = ((1 - self.alpha_bar_t[t - 1]) / (1 - self.alpha_bar_t[t])) * (1 - self.alpha_t[t])
                sigma_t = torch.sqrt(beta_t)

                x_t = (1 / torch.sqrt(self.alpha_t[t])) * (
                    x_t - (((1 - self.alpha_t[t]) / (torch.sqrt(1 - self.alpha_bar_t[t]))) * predicted_noise)
                    ) + sigma_t * z
        
        return x_t


