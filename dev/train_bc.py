import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser
import random

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm
import numpy as np

from east_dataset import EASTDataset
from dataset import SceneTextDataset, BoostCampDataset
from model import EAST
import wandb
from PIL import ImageFile 
ImageFile.LOAD_TRUNCATED_IMAGES = True

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--train_data_dir', type=str,
                        default='/opt/ml/input/data/ICDAR19')
    parser.add_argument('--val_data_dir', type=str,
                        default='/opt/ml/input/data/valid_bc')
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=41)
    parser.add_argument('--save_interval', type=int, default=2)
    parser.add_argument('--seed', type=int, default=16, help="random seed (default: 16)")
    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args
    


def do_training(train_data_dir, val_data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, seed):
    seed_everything(args.seed)

    train_dataset = SceneTextDataset(train_data_dir, split='mlt19_train', image_size=1024, crop_size=512)
    val_dataset = BoostCampDataset(val_data_dir, image_size=1024, crop_size=512)

    train_dataset = EASTDataset(train_dataset)
    val_dataset = EASTDataset(val_dataset)

    num_batches = math.ceil(len(train_dataset) / batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)
    config = wandb.config
    config.learning_rate = args.learning_rate

    wandb.watch(model)
    best_loss = 10000 
    model.train()
    for epoch in range(max_epoch):
        epoch_loss, epoch_start = 0, time.time()
        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Epoch {}]'.format(epoch + 1))
                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                pbar.update(1)
                train_dict = {
                    'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss'], 'Mean Loss': extra_info['cls_loss']+extra_info['angle_loss']+extra_info['iou_loss']
                }

                pbar.set_postfix(train_dict)
                wandb.log(train_dict)

        scheduler.step()

        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))

        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_fpath = osp.join(model_dir, f'latest_{epoch+1}.pth')
            torch.save(model.state_dict(), ckpt_fpath)
            #Validation
            print(f"Start validation #{epoch+1}")
            print("Calculating validation results...")
            epoch_loss, epoch_start = 0, time.time()
            with torch.no_grad():
                model.eval()
                for img, gt_score_map, gt_geo_map, roi_mask in val_loader:
                    loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                    loss_val = loss.item()
                    epoch_loss += loss_val
                    val_dict = {
                        'VAL_Cls loss': extra_info['cls_loss'], 'VAL_Angle loss': extra_info['angle_loss'],
                        'VAL_IoU loss': extra_info['iou_loss'], 'VAL_Mean loss': extra_info['cls_loss']+extra_info['angle_loss']+ extra_info['iou_loss']
                    }
                    wandb.log(val_dict)
                print('Valid loss: {:.4f} | Elapsed time: {}'.format(
                epoch_loss , timedelta(seconds=time.time() - epoch_start)))
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    if not osp.exists(model_dir):
                        os.makedirs(model_dir)

                    ckpt_fpath = osp.join(model_dir, f'valid_{epoch+1}.pth')
                    torch.save(model.state_dict(), ckpt_fpath)




def main(args):
    do_training(**args.__dict__)


if __name__ == '__main__':
    args = parse_args()
    wandb.init(project="ocr", entity="passion-ate")
    wandb.run.name = (
        f"mlt19_All"
    )
    wandb.config.update(args)
    main(args)

