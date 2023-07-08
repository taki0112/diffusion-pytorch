import argparse
from utils import *
from GALIP import run_fn

"""
count params
G network parameters :  51,830,211
D network parameters :  30,806,021
Total network parameters :  82,636,232
"""

def parse_args():
    desc = "Pytorch implementation of GALIP"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='[train, test]')
    parser.add_argument('--dataset', type=str, default='coco_2017', help='dataset_name')
    # celeba_hq_text
    # coco_2017
    parser.add_argument('--epoch', type=int, default=3000, help='The total epoch')
    parser.add_argument('--iteration', type=int, default=1000000, help='The total iterations')
    parser.add_argument('--img_size', type=int, default=256, help='The size of image')
    parser.add_argument('--batch_size', type=int, default=64, help='batch sizes for each gpus')
    parser.add_argument('--mixed_flag', type=str2bool, default=True, help='Mixed Precision Flag')
    # single = 16

    # StyleGAN paraeter
    parser.add_argument("--g_lr", type=float, default=0.0001, help="g learning rate")
    parser.add_argument("--d_lr", type=float, default=0.0004, help="d learning rate")

    parser.add_argument('--print_freq', type=int, default=5000, help='The number of image_print_freq')
    parser.add_argument('--save_freq', type=int, default=50000, help='The number of ckpt_save_freq')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory name to save the samples_prev on training')

    return check_args(parser.parse_args())


"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --sample_dir
    check_folder(args.sample_dir)

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one', flush=True)

    return args

"""main"""
def main():

    args = vars(parse_args())

    # run
    multi_gpu_run(ddp_fn=run_fn, args=args)

if __name__ == '__main__':
    main()