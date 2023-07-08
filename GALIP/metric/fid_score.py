import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from torchvision import models
import torch.distributed as dist
import math
from tqdm import tqdm
from torchvision import transforms
from scipy import linalg
import pickle, os
from torch.nn.functional import adaptive_avg_pool2d

class dummy_context_mgr():
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False

class GatherLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


class InceptionV3_(nn.Module):
    def __init__(self):
        super().__init__()
        inception = models.inception_v3(weights='DEFAULT')
        # pretrained=True -> weights=Inception_V3_Weights.IMAGENET1K_V1
        # weights='DEFAULT' or weights='IMAGENET1K_V1'
        self.block1 = nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.block2 = nn.Sequential(
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.block3 = nn.Sequential(
            inception.Mixed_5b, inception.Mixed_5c,
            inception.Mixed_5d, inception.Mixed_6a,
            inception.Mixed_6b, inception.Mixed_6c,
            inception.Mixed_6d, inception.Mixed_6e)
        self.block4 = nn.Sequential(
            inception.Mixed_7a, inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x.view(x.size(0), -1)

class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 mixed_precision=False,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False):
        """Build pretrained InceptionV3

        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, normalizes the input to the statistics the pretrained
            Inception network expects
        requires_grad : bool
            If true, parameters of the model require gradient. Possibly useful
            for finetuning the network
        """
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        inception = models.inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps

        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)

        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.upsample(x, size=(299, 299), mode='bilinear', align_corners=True)

        if self.normalize_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp

def extract_real_feature(data_loader, inception, device, t2i=False):
    feats = []

    if t2i:
        for img, tokens, txt in tqdm(data_loader):
            img = img.to(device)
            feat = inception(img)

            feats.append(feat)
    else:
        for img in tqdm(data_loader):
            img = img.to(device)
            feat = inception(img)

            feats.append(feat)

    feats = gather_feats(feats)

    return feats

def normalize_fake_img(imgs):
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    imgs = (imgs + 1) / 2  # -1 ~ 1 to 0~1
    imgs = torch.clamp(imgs, 0, 1)
    imgs = transforms.Resize(size=[299, 299], interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)(imgs),
    imgs = transforms.Normalize(mean=mean, std=std)(imgs)
    """

    norm = transforms.Compose([
        transforms.Normalize((-1, -1, -1), (2, 2, 2)), # (x - (-1)) / 2 = (x + 1) / 2
        transforms.Resize((299, 299)),
    ])

    x = norm(imgs)

    return x

def gather_feats(feats):
    feats = torch.cat(feats, dim=0)
    feats = torch.cat(GatherLayer.apply(feats), dim=0)
    feats = feats.detach().cpu().numpy()

    return feats

def extract_fake_feature(generator, inception, num_gpus, device, latent_dim, fake_samples=50000, batch_size=16):
    num_batches = int(math.ceil(float(fake_samples) / float(batch_size * num_gpus)))
    feats = []
    for _ in tqdm(range(num_batches)):
        z = [torch.randn([batch_size, latent_dim], device=device)]
        fake_img = generator(z)

        fake_img = normalize_fake_img(fake_img)

        feat = inception(fake_img)

        feats.append(feat)

    feats = gather_feats(feats)

    return feats

def extract_fake_feature_t2i(data_loader, generator, inception, clip_text, device, latent_dim=100, mixed_flag=False):
    # with torch.cuda.amp.autocast() if mixed_flag else dummy_context_mgr() as mpc:
    with torch.no_grad():
        feats = []
        try:
            for img, tokens, txt in tqdm(data_loader):
                    # pre-process
                    tokens = tokens.to(device)
                    sent_emb, word_emb = clip_text(tokens)  # [bs, 512], [bs, 77, 512]
                    sent_emb = sent_emb.detach()

                    # make fake_img
                    noise = torch.randn([sent_emb.shape[0], latent_dim]).to(device)
                    fake_img = generator(noise, sent_emb)
                    fake_img = fake_img.float()
                    fake_img = torch.clamp(fake_img, -1., 1.)
                    fake_img = torch.nan_to_num(fake_img, nan=-1.0, posinf=1.0, neginf=-1.0)

                    # get features of inception
                    fake_img = normalize_fake_img(fake_img)
                    feat = inception(fake_img)

                    # galip
                    pred = feat[0]
                    if pred.shape[2] != 1 or pred.shape[3] != 1:
                        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
                    pred = pred.squeeze(-1).squeeze(-1)
                    feats.append(pred)

        except IndexError:
            pass

        feats = gather_feats(feats)

    return feats

def get_statistics(feats):
    mu = np.mean(feats, axis=0)
    cov = np.cov(feats, rowvar=False)

    return mu, cov

def frechet_distance(mu, cov, mu2, cov2):
    cc, _ = linalg.sqrtm(np.dot(cov, cov2), disp=False)
    dist = np.sum((mu -mu2)**2) + np.trace(cov + cov2 - 2*cc)
    return np.real(dist)

@torch.no_grad()
def calculate_fid(data_loader, generator_model, inception_model, dataset_name, rank, device,
                  latent_dim, fake_samples=50000, batch_size=16):

    num_gpus = torch.cuda.device_count()

    generator_model = generator_model.eval()
    inception_model = inception_model.eval()

    pickle_name = '{}_mu_cov.pickle'.format(dataset_name)
    cache = os.path.exists(pickle_name)

    if cache:
        with open(pickle_name, 'rb') as f:
            real_mu, real_cov = pickle.load(f)
    else:
        real_feats = extract_real_feature(data_loader, inception_model, device=device)
        real_mu, real_cov = get_statistics(real_feats)

        if rank == 0:
            with open(pickle_name, 'wb') as f:
                pickle.dump((real_mu, real_cov), f, protocol=pickle.HIGHEST_PROTOCOL)


    fake_feats = extract_fake_feature(generator_model, inception_model, num_gpus, device, latent_dim, fake_samples, batch_size)
    fake_mu, fake_cov = get_statistics(fake_feats)

    fid = frechet_distance(real_mu, real_cov, fake_mu, fake_cov)

    return fid

@torch.no_grad()
def calculate_fid_t2i(data_loader, generator, inception, clip_text, dataset_name, device,
                      latent_dim=100, mixed_flag=False):
    # coco: 5000

    generator = generator.eval()
    inception = inception.eval()
    clip_text = clip_text.eval()

    stats_path = '{}_fid_stats.npz'.format(dataset_name)
    x = np.load(stats_path)
    real_mu, real_cov = x['mu'], x['sigma']


    fake_feats = extract_fake_feature_t2i(data_loader, generator, inception, clip_text, device, latent_dim, mixed_flag)
    fake_mu, fake_cov = get_statistics(fake_feats)

    fid = frechet_distance(real_mu, real_cov, fake_mu, fake_cov)

    return fid