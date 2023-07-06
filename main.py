import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet, linear_beta_schedule, cosine_beta_schedule
import logging
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, objective='ddpm', schedule='linear', device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.objective = objective

        self.beta = self.prepare_noise_schedule(schedule).to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self, schedule):
        if schedule == 'linear':
            return linear_beta_schedule(self.noise_steps)
        else:
            return cosine_beta_schedule(self.noise_steps)


    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        z = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * z, z

    def sample_timesteps(self, n):
        t = torch.randint(low=1, high=self.noise_steps, size=(n,))
        return t

    def tensor_to_image(self, x):
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

    def sample(self, model, n):
        # reverse process
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps))):
                t = (torch.ones(n, dtype=torch.long) * i).to(self.device)

                alpha = self.alpha[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                alpha_hat_prev = self.alpha_hat[t-1][:, None, None, None]
                beta_tilde = beta * (1 - alpha_hat_prev) / (1 - alpha_hat) # similar to beta

                predicted_noise = model(x, t)
                noise = torch.randn_like(x)

                if self.objective == 'ddpm':
                    predict_x0 = 0
                    direction_point = 1 / torch.sqrt(alpha) * (x - (beta / (torch.sqrt(1 - alpha_hat))) * predicted_noise)
                    random_noise = torch.sqrt(beta_tilde) * noise

                    x = predict_x0 + direction_point + random_noise
                else:
                    predict_x0 = alpha_hat_prev * (x - torch.sqrt(1 - alpha_hat) * predicted_noise) / torch.sqrt(alpha_hat)
                    direction_point = torch.sqrt(1 - alpha_hat_prev) * predicted_noise
                    random_noise = 0

                    x = predict_x0 + direction_point + random_noise

        model.train()
        return torch.clamp(x, -1.0, 1.0)


def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet(device=device).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("logs", args.run_name))
    l = len(dataloader)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, images in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        sampled_images = diffusion.sample(model, n=images.shape[0])
        save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.png"))
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.epochs = 100
    args.batch_size = 16
    args.image_size = 64
    args.objective = 'ddpm'
    args.schedule = 'linear'
    args.dataset_path = "./dataset/cat"
    args.device = "cuda"
    args.lr = 3e-4

    args.run_name = "diffusion_{}_{}".format(args.objective, args.schedule)
    train(args)


if __name__ == '__main__':
    launch()