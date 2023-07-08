import torch.utils.data
from ops import *
from utils import *
import time
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torchvision
from functools import partial
import torch.nn.functional as F

print = partial(print, flush=True)

from metric.fid_score import InceptionV3, calculate_fid_t2i
from networks import *

def run_fn(rank, args, world_size):
    device = torch.device('cuda', rank)
    torch.backends.cudnn.benchmark = True

    model = GALIP(args, world_size)
    model.build_model(rank, device)
    model.train_model(rank, device)

class GALIP():
    def __init__(self, args, NUM_GPUS):
        super(GALIP, self).__init__()

        """ Model """
        self.model_name = 'GALIP'
        self.phase = args['phase']
        self.NUM_GPUS = NUM_GPUS


        """ Training parameters """
        self.img_size = args['img_size']
        self.batch_size = args['batch_size']
        self.global_batch_size = self.batch_size * self.NUM_GPUS
        self.epoch = args['epoch']
        if self.epoch != 0:
            self.iteration = None
        else:
            self.iteration = args['iteration']
        self.mixed_flag = args['mixed_flag']
        self.growth_interval = 2000
        self.scaler_min = 64

        """ Network parameters """
        self.style_dim = 100
        self.g_lr = args['g_lr']
        self.d_lr = args['d_lr']

        """ Print parameters """
        self.print_freq = args['print_freq']
        self.save_freq = args['save_freq']
        self.log_template = 'step [{}/{}]: elapsed: {:.2f}s, BEST_FID: {:.2f}'

        """ Dataset Path """
        self.dataset_name = args['dataset']
        self.val_dataset_name = self.dataset_name + '_val'
        dataset_path = './dataset'
        self.dataset_path = os.path.join(dataset_path, self.dataset_name)
        self.val_dataset_path = os.path.join(dataset_path, self.val_dataset_name)

        """ Directory """
        self.checkpoint_dir = args['checkpoint_dir']
        self.result_dir = args['result_dir']
        self.log_dir = args['log_dir']
        self.sample_dir = args['sample_dir']

        self.sample_dir = os.path.join(self.sample_dir, self.model_dir)
        check_folder(self.sample_dir)
        self.checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)
        check_folder(self.checkpoint_dir)
        self.log_dir = os.path.join(self.log_dir, self.model_dir)
        check_folder(self.log_dir)

    ##################################################################################
    # Model
    ##################################################################################
    def build_model(self, rank, device):
        """ Init process """
        build_init_procss(rank, world_size=self.NUM_GPUS, device=device)

        """ Dataset Load """
        dataset = ImageTextDataset(dataset_path=self.dataset_path, img_size=self.img_size)
        self.dataset_num = dataset.__len__()
        self.iteration = self.epoch * self.dataset_num // self.global_batch_size
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=4,
                                             sampler=distributed_sampler(dataset, rank=rank, num_replicas=self.NUM_GPUS, shuffle=True),
                                             drop_last=True, pin_memory=True)
        self.dataset_iter = infinite_iterator(loader)

        """ For FID """
        self.fid_dataset = ImageTextDataset(dataset_path=self.val_dataset_path, img_size=299, imagenet_normalization=True)
        self.fid_loader = torch.utils.data.DataLoader(self.fid_dataset, batch_size=5, num_workers=4,
                                             sampler=distributed_sampler(dataset, rank=rank, num_replicas=self.NUM_GPUS, shuffle=False),
                                             drop_last=False, pin_memory=True)
        self.inception = InceptionV3(mixed_precision=self.mixed_flag).to(device)
        requires_grad(self.inception, False)

        """ Pretrain Model Load """
        self.clip = clip.load('ViT-B/32')[0].eval().to(device)
        self.clip_img = CLIP_IMG_ENCODER(self.clip).to(device)
        self.clip_text = CLIP_TXT_ENCODER(self.clip).to(device)

        requires_grad(self.clip_img, False)
        requires_grad(self.clip_text, False)

        self.clip_img.eval()
        self.clip_text.eval()


        """ Network """
        if self.mixed_flag:
            self.scaler_G = torch.cuda.amp.GradScaler(growth_interval=self.growth_interval)
            self.scaler_D = torch.cuda.amp.GradScaler(growth_interval=self.growth_interval)
        else:
            self.scaler_G = None
            self.scaler_D = None
        self.generator = NetG(imsize=self.img_size, CLIP=self.clip, nz=self.style_dim, mixed_precision=self.mixed_flag).to(device)
        self.discriminator = NetD(imsize=self.img_size, mixed_precision=self.mixed_flag).to(device)
        self.predictor = NetC(mixed_precision=self.mixed_flag).to(device)


        """ Optimizer """
        D_params = list(self.discriminator.parameters()) + list(self.predictor.parameters())
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.g_lr, betas=(0.0, 0.9), eps=1e-08)
        self.d_optimizer = torch.optim.Adam(D_params, lr=self.d_lr, betas=(0.0, 0.9), eps=1e-08)

        """ Distributed Learning """
        self.generator = dataparallel_and_sync(self.generator, rank)
        self.discriminator = dataparallel_and_sync(self.discriminator, rank)
        self.predictor = dataparallel_and_sync(self.predictor, rank)

        """ Checkpoint """
        self.ckpt_dict= {
                            'generator': self.generator.state_dict(),
                            'discriminator': self.discriminator.state_dict(),
                            'predictor' : self.predictor.state_dict(),
                            'g_optimizer': self.g_optimizer.state_dict(),
                            'd_optimizer':self.d_optimizer.state_dict()
                        },

        latest_ckpt_name, start_iter = find_latest_ckpt(self.checkpoint_dir)
        if latest_ckpt_name is not None:
            if rank == 0:
                print('Latest checkpoint restored!! ', latest_ckpt_name)
                print('start iteration : ', start_iter)
            self.start_iteration = start_iter

            latest_ckpt = os.path.join(self.checkpoint_dir, latest_ckpt_name)
            ckpt = torch.load(latest_ckpt, map_location=device)

            self.generator.load_state_dict(ckpt["generator"])
            self.discriminator.load_state_dict(ckpt['discriminator'])
            self.predictor.load_state_dict(ckpt['predictor'])
            self.g_optimizer.load_state_dict(ckpt["g_optimizer"])
            self.d_optimizer.load_state_dict(ckpt["d_optimizer"])

        else:
            if rank == 0:
                print('Not restoring from saved checkpoint')
            self.start_iteration = 0

    def g_train_step(self, real_img, tokens, device=torch.device('cuda')):
        self.generator.train()
        self.discriminator.train()
        self.predictor.train()

        # step 0: pre-process
        with torch.cuda.amp.autocast() if self.mixed_flag else dummy_context_mgr() as mpc:
            with torch.no_grad():
                sent_emb, word_emb = self.clip_text(tokens) # [bs, 512], [bs, 77, 512]
                word_emb = word_emb.detach()
                sent_emb = sent_emb.detach()

            # synthesize fake images
            noise = torch.randn([self.batch_size, self.style_dim]).to(device)
            fake_img = self.generator(noise, sent_emb)
            CLIP_fake, fake_emb = self.clip_img(fake_img)

            # loss
            fake_feats = self.discriminator(CLIP_fake)
            output = self.predictor(fake_feats, sent_emb)
            text_img_sim = torch.cosine_similarity(fake_emb, sent_emb).mean()
            loss = -output.mean() - 4.0 * text_img_sim

        apply_gradients(loss, self.g_optimizer, self.mixed_flag, self.scaler_G, self.scaler_min)

        return loss, sent_emb

    def d_train_step(self, real_img, tokens, device=torch.device('cuda')):
        self.generator.train()
        self.discriminator.train()
        self.predictor.train()

        # step 0: pre-process
        with torch.cuda.amp.autocast() if self.mixed_flag else dummy_context_mgr() as mpc:
            with torch.no_grad():
                sent_emb, word_emb = self.clip_text(tokens) # [bs, 512], [bs, 77, 512]
                word_emb = word_emb.detach()
                sent_emb = sent_emb.detach()


            # loss
            real_img = real_img.requires_grad_()
            sent_emb = sent_emb.requires_grad_()
            word_emb = word_emb.requires_grad_()

            # predict real
            CLIP_real, real_emb = self.clip_img(real_img) # [bs, 3, 768, 7, 7], [bs, 512]
            real_feats = self.discriminator(CLIP_real) # [bs, 512, 7, 7]
            pred_real, errD_real = predict_loss(self.predictor, real_feats, sent_emb, negtive=False)

            # predict mismatch
            mis_sent_emb = torch.cat((sent_emb[1:], sent_emb[0:1]), dim=0).detach()
            _, errD_mis = predict_loss(self.predictor, real_feats, mis_sent_emb, negtive=True)

            # synthesize fake images
            noise = torch.randn([self.batch_size, self.style_dim]).to(device)
            fake_img = self.generator(noise, sent_emb)
            CLIP_fake, fake_emb = self.clip_img(fake_img)
            fake_feats = self.discriminator(CLIP_fake.detach())
            _, errD_fake = predict_loss(self.predictor, fake_feats, sent_emb, negtive=True)

        if self.mixed_flag:
            errD_MAGP = MA_GP_MP(CLIP_real, sent_emb, pred_real, self.scaler_D)
        else:
            errD_MAGP = MA_GP_FP32(CLIP_real, sent_emb, pred_real)

        with torch.cuda.amp.autocast() if self.mixed_flag else dummy_context_mgr() as mpc:
            loss = errD_real + (errD_fake + errD_mis) / 2.0 + errD_MAGP

        apply_gradients(loss, self.d_optimizer, self.mixed_flag, self.scaler_D, self.scaler_min)

        return loss

    def train_model(self, rank, device):
        start_time = time.time()
        fid_start_time = time.time()

        # setup tensorboards
        train_summary_writer = SummaryWriter(self.log_dir)

        # start training
        if rank == 0:
            print()
            print(self.dataset_path)
            print("Dataset number : ", self.dataset_num)
            print("GPUs : ", self.NUM_GPUS)
            print("Each batch size : ", self.batch_size)
            print("Global batch size : ", self.global_batch_size)
            print("Target image size : ", self.img_size)
            print("Print frequency : ", self.print_freq)
            print("Save frequency : ", self.save_freq)
            print("PyTorch Version :", torch.__version__)
            print('max_steps: {}'.format(self.iteration))
            print()
        losses = {'g_loss': 0.0, 'd_loss': 0.0}

        fid_dict = {'metric/fid': 0.0, 'metric/best_fid': 0.0, 'metric/best_fid_iter': 0}
        fid = 0
        best_fid = 1000
        best_fid_iter = 0

        for idx in range(self.start_iteration, self.iteration):
            iter_start_time = time.time()

            image, tokens, text = next(self.dataset_iter)
            image = image.to(device)
            tokens = tokens.to(device)
            # text = text.to(device)

            if idx == 0:
                if rank == 0:
                    print("count params")
                    g_params = count_parameters(self.generator)
                    d_params = count_parameters(self.discriminator) + count_parameters(self.predictor)
                    g_B, g_M = convert_to_billion_and_million(g_params)
                    d_B, d_M = convert_to_billion_and_million(d_params)

                    t_B = g_B + d_B
                    t_M = g_M + d_M

                    print("G network parameters : {}B, {}M".format(g_B, g_M))
                    print("D network parameters : {}B, {}M".format(d_B, d_M))
                    print("Total network parameters : {}B, {}M".format(t_B, t_M))
                    print()

            loss = self.d_train_step(image, tokens, device=device)

            losses['d_loss'] = loss

            loss, text_embed = self.g_train_step(image, tokens, device)
            losses['g_loss'] = loss

            losses = reduce_loss_dict(losses)
            losses = dict_to_numpy(losses, python_value=True)

            if np.mod(idx, self.print_freq) == 0 or idx == self.iteration - 1 :
                if rank == 0:
                    print("calculate fid ...")
                    fid_start_time = time.time()

                fid = calculate_fid_t2i(self.fid_loader, self.generator, self.inception, self.clip_text, self.val_dataset_name,
                                        device=device, latent_dim=self.style_dim)

                if rank == 0:
                    fid_end_time = time.time()
                    fid_elapsed = fid_end_time - fid_start_time
                    print("calculate fid finish: {:.2f}s".format(fid_elapsed))
                    if fid < best_fid:
                        print("BEST FID UPDATED")
                        best_fid = fid
                        best_fid_iter = idx
                        self.torch_save(idx, fid)

                        fid_dict['metric/best_fid'] = best_fid
                        fid_dict['metric/best_fid_iter'] = best_fid_iter
                    fid_dict['metric/fid'] = fid


            if rank == 0:
                # save to tensorboard

                for k, v in losses.items():
                    train_summary_writer.add_scalar(k, v, global_step=idx)

                if np.mod(idx, self.print_freq) == 0 or idx == self.iteration - 1:
                    train_summary_writer.add_scalar('fid', fid, global_step=idx)

                if np.mod(idx + 1, self.print_freq) == 0:
                    with torch.no_grad():
                        batch_size = text_embed.shape[0]

                        noise = torch.randn([batch_size, self.style_dim]).to(device)
                        self.generator.eval()
                        fake_img = self.generator(noise, text_embed)
                        fake_img = torch.clamp(fake_img, -1.0, 1.0)

                        partial_size = int(batch_size ** 0.5)

                        # resize
                        fake_img = F.interpolate(fake_img, size=256, mode='bicubic', align_corners=True)
                        torchvision.utils.save_image(fake_img, './{}/fake_{:06d}.png'.format(self.sample_dir, idx + 1),
                                                     nrow=partial_size,
                                                     normalize=True, range=(-1, 1))
                        text_path = './{}/fake_{:06d}.txt'.format(self.sample_dir, idx+1)
                        with open(text_path, 'w') as f:
                            f.write('\n'.join(text))
                        # normalize = set to the range (0, 1) by range(min, max)

                elapsed = time.time() - iter_start_time
                print(self.log_template.format(idx, self.iteration, elapsed, best_fid))

            dist.barrier()

        if rank == 0:
            # save model for final step
            self.torch_save(self.iteration, fid)

            print("LAST FID: ", fid)
            print("BEST FID: {}, {}".format(best_fid, best_fid_iter))
            print("Total train time: %4.4f" % (time.time() - start_time))

        dist.barrier()

    def torch_save(self, idx, fid=0):
        fid_int = int(fid)
        torch.save(
            self.ckpt_dict,
            os.path.join(self.checkpoint_dir, 'iter_{}_fid_{}.pt'.format(idx, fid_int))
        )

    @property
    def model_dir(self):
        return "{}_{}_{}_bs{}_{}GPUs_Mixed{}".format(self.model_name, self.dataset_name, self.img_size, self.batch_size, self.NUM_GPUS, self.mixed_flag)