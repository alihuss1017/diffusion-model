import dataset
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import ImageGrid
import torch
import asyncio 

max_time_steps = 50
batch_size = 64
train_split = 0.7
val_split = 0.3
subset_size = 50000
async def load_images():
    builder = dataset.BuildLoaders(train_split, val_split, batch_size, subset_size)
    train_loader, _ = builder.build()

    images = next(iter(train_loader)) # (64, 3, 64, 64)
    normalized_images = (images - 127.5) / 127.5
    
    return normalized_images

async def compute_alphas():
    t = torch.arange(0, max_time_steps, 1)
    s = 0.008
    f_t = torch.cos((((t / t[-1]) + s) / (1 + s)) * (torch.pi / 2)) ** 2 # (50, )
    alpha_t = f_t / f_t[0] # (50, )
    return alpha_t

def sample_alphas(alpha_t):
    indices = torch.randint(0, max_time_steps, (batch_size, 1)) # (batch_size, 1)
    alphas = alpha_t[indices] # (batch size, 1)
    alphas = alphas.reshape(batch_size, 1, 1, 1) # (batch_size, 1, 1, 1)
    return indices, alphas

async def compute_noisy_images():
    images_task = asyncio.create_task(load_images())
    alphas_task = asyncio.create_task(compute_alphas())

    normalized_images, alpha_t = await asyncio.gather(images_task, alphas_task)

    indices, alphas = sample_alphas(alpha_t)
    eps = torch.normal(0, 1, size = normalized_images.shape)
    noisy_images = torch.sqrt(alphas) * normalized_images + torch.sqrt(1 - alphas) * eps
    return indices, noisy_images


if __name__ == "__main__":
    indices, noisy_images = asyncio.run(compute_noisy_images())
    sample_variations = noisy_images[:36].permute(0, 2, 3, 1)
    sample_variations = (sample_variations + 1) / 2
    fig = plt.figure(figsize=(12., 12.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(6, 6),  # creates 2x2 grid of Axes
                    axes_pad=0.4,  # pad between Axes in inch.
                    )

    for ax, idx, im in zip(grid, indices, sample_variations):
        # Iterating over the grid returns the Axes.
        ax.set_title(f't = {idx.item()}')
        ax.imshow(im)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Forward Diffusion Visualization At Different Time Steps")
    plt.show()

