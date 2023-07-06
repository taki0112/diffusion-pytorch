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

        self.beta = self.prepare_noise_schedule(schedule, beta_start, beta_end).to(device)

        """
        Step 1. 
        
        self.alpha = ?
        self.alpha_hat = ?
        
        """


    def prepare_noise_schedule(self, schedule, beta_start, beta_end):
        if schedule == 'linear':
            return linear_beta_schedule(self.noise_steps, beta_start, beta_end)
        else:
            return cosine_beta_schedule(self.noise_steps)

    def sample_timesteps(self, n):
        """
        Step 2.
        n개의 랜덤한 timestep을 샘플링 하세요. range = [1, self.noise_steps]

        :param n: int
        :return: [n, ] shape을 갖고있을것입니다.

        주의사항: timestep이니까, 값은 int형이어야 합니다.

        """
        return

    def noise_images(self, x, t):
        """
        Step 3.
        forward process를 작성하세요.
        -> 이미지에 noise를 입히는 과정입니다.

        return은 노이즈를 입힌 이미지와, 입혔던 노이즈를 리턴하세요 !! 총 2개입니다.

        :param x: [n, 3, img_size, img_size]
        :param t: [n, ]
        :return: [n, 3, img_size, img_size], [n, 3, img_size, img_size]

        """
        return


    def sample(self, model, n):
        """
        Step 5. 마지막!
        reverse process를 완성하세요.

        :param model: Unet
        :param n: batch_size
        :return: x: [n, 3, img_size, img_size]
        """
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            """
            (1) T스텝에서 부터 denoise하는것이기때문에, 가우시안 noise를 하나 만드세요.
            (2) T (self.noise_steps)부터 denoise하는 구문을 만드세요. 
                hint: T, T-1, T-2, ... , 3, 2, 1 이런식으로 t가 나와야겠죠 ?
            (3) t에 해당하는 alpha_t, beta_t, alpha_hat_t, alpha_hat_(t-1), beta_tilde를 만드세요.
            
            (4) (1)의 noise와 (2)의 t를 모델에 넣어서, noise를 predict하세요.
            (5) predict한 noise를 가지고, ddpm과 ddim sampling를 작성하세요.
            
            """

        model.train()
        return torch.clamp(x, -1.0, 1.0)


def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet(device=device).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("logs", args.run_name))
    l = len(dataloader)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, images in enumerate(pbar):
            images = images.to(device)
            """
            Step 4.
            학습코드를 작성해보세요.
            다음 hint를 참고하여 작성하면됩니다.
            
            hint:
            (1) timestep을 샘플링 하세요.
            (2) 해당 timestep t에 대응되는 노이즈 입힌 이미지를 만드세요.
            (3) 모델에 넣어서, 노이즈를 predict 하세요.
            (4) 적절한 loss를 선택하세요. (L1 or L2)
            """

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("diffusion loss", loss.item(), global_step=epoch * l + i)

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
    args.device = "cpu"
    args.lr = 3e-4

    args.run_name = "diffusion_{}_{}".format(args.objective, args.schedule)
    train(args)


if __name__ == '__main__':
    launch()