import argparse
import gc
import logging
import os
import sys

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from dataloaders.dataset import *
from monai.losses import DiceCELoss
from networks.critic import Discriminator
from networks.net_factory import net_factory
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from utils import ramps, losses, test_patch_sup
from utils.losses import loss_diff1, loss_mask, loss_diff2, disc_loss, gen_loss

from dataloaders.msd_brats import BraTS

from utils.dice_msd_brats_v1 import EDiceLoss


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='MSD_BRATS', help='dataset_name')
parser.add_argument('--root_path', type=str, default='/data/himashi/Co-BioNet_MC/', help='Name of Dataset')
parser.add_argument('--exp', type=str, default='Co_BioNet_SEMI', help='exp_name')
parser.add_argument('--model', type=str, default='vnet', help='model_name')
parser.add_argument('--max_iteration', type=int, default=10000, help='maximum iteration to train')
parser.add_argument('--max_samples', type=int, default=387, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=2, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size of labeled data per gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=117, help='trained samples')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--consistency', type=float, default=1.0, help='consistency_weight')
parser.add_argument('--consistency_rampup', type=float, default=70.0, help='consistency_rampup')
parser.add_argument('--lamda', type=float, default=1.0, help='weight to balance all losses')
parser.add_argument('--mu', type=float, default=0.01, help='weight to balance generator adversarial loss')
parser.add_argument('--t_m', type=float, default=0.2, help='mask threashold')
args = parser.parse_args()

snapshot_path = args.root_path + "model/{}_{}_{}_labeled/{}".format(args.dataset_name, args.exp, args.labelnum,
                                                                    args.model)

num_classes = 4

patch_size = (128, 128, 96)
args.root_path = '/data/himashi/Co-BioNet_MC/data/MSD_BRATS/'
args.max_samples = 387
train_data_path = args.root_path

labeled_bs = args.labeled_bs
max_iterations = args.max_iteration
base_lr = args.base_lr

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

if __name__ == "__main__":
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    model_1 = net_factory(net_type=args.model, in_chns=4, class_num=num_classes - 1, mode="train")
    model_2 = net_factory(net_type=args.model, in_chns=4, class_num=num_classes - 1, mode="train")
    model_1 = model_1.cuda()
    model_2 = model_2.cuda()

    critic_1 = Discriminator()
    critic_2 = Discriminator()
    critic_1 = critic_1.cuda()
    critic_2 = critic_2.cuda()

    db_train = BraTS(base_dir=train_data_path, split='train', patch_size=patch_size)

    db_val = BraTS(base_dir=train_data_path, split='val', patch_size=patch_size)

    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=2, pin_memory=True)
    valloader = DataLoader(db_val, batch_size=1, num_workers=2, pin_memory=True)

    optimizer_1 = optim.SGD(model_1.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer_2 = optim.SGD(model_2.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    dis_optimizer_1 = torch.optim.AdamW(critic_1.parameters(), lr=1e-4)
    dis_optimizer_2 = torch.optim.AdamW(critic_2.parameters(), lr=1e-4)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    iter_num = 0
    best_dice_1 = 0
    best_dice_2 = 0
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr
    CE = torch.nn.BCELoss()
    iterator = tqdm(range(max_epoch), ncols=70)
    criterion = EDiceLoss().cuda()

    scheduler_1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_1, T_max=max_epoch)
    scheduler_2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_2, T_max=max_epoch)
    c_scheduler_1 = torch.optim.lr_scheduler.CosineAnnealingLR(dis_optimizer_1, T_max=max_epoch)
    c_scheduler_2 = torch.optim.lr_scheduler.CosineAnnealingLR(dis_optimizer_2, T_max=max_epoch)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'].float(), sampled_batch['label'].float()
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            labeled_gt_set = label_batch[:labeled_bs, ...]
            labeled_set = volume_batch[:labeled_bs, ...]
            unlabeled_set = volume_batch[labeled_bs:, ...]

            # Train Model 1
            model_1.train()

            l_outputs_np_1 = model_1(labeled_set)
            u_outputs_np_1 = model_1(unlabeled_set)

            # Train Model 2
            model_2.train()

            u_outputs_np_2 = model_2(unlabeled_set)
            l_outputs_np_2 = model_2(labeled_set)

            y_prob_1 = torch.sigmoid(l_outputs_np_1)
            y_all_1 = torch.sigmoid(model_1(volume_batch))
            y_sig_1_u = torch.sigmoid(u_outputs_np_1)

            loss_seg_dice_1 = criterion(y_prob_1, labeled_gt_set)

            y_prob_2 = torch.sigmoid(l_outputs_np_2)
            y_all_2 = torch.sigmoid(model_2(volume_batch))
            y_sig_2_u = torch.sigmoid(u_outputs_np_2)

            loss_seg_dice_2 = criterion(y_prob_2, labeled_gt_set)

            loss_sup_1 = loss_seg_dice_1
            loss_sup_2 = loss_seg_dice_2
            iter_num = iter_num + 1
            consistency_weight = get_current_consistency_weight(iter_num // 100)

            loss_dif_1 = loss_diff1(y_sig_1_u, y_sig_2_u)
            critic_segs_1 = torch.sigmoid(critic_2(y_sig_2_u))
            masked_loss_1 = loss_mask(y_sig_1_u, y_sig_2_u, critic_segs_1, args.t_m)

            g_critic_segs_1_1 = torch.sigmoid(critic_1(y_all_1))
            g_critic_segs_1_2 = torch.sigmoid(critic_1(labeled_gt_set.float()))

            loss_adversarial_gen_1 = gen_loss(g_critic_segs_1_1)
            loss_adversarial_1 = disc_loss(g_critic_segs_1_1, g_critic_segs_1_2)

            loss_unsup_1 = loss_dif_1 + masked_loss_1 + args.mu * loss_adversarial_gen_1
            loss_1 = args.lamda * loss_sup_1 + consistency_weight * loss_unsup_1

            optimizer_1.zero_grad()
            loss_1.backward()
            optimizer_1.step()
            logging.info(
                'M1 iteration %d : loss : %03f, loss_sup: %03f, loss_diff: %03f, loss_masked: %03f, loss_adv: %03f, consistency_weight: %03f' % (
                    iter_num, loss_1, loss_sup_1, loss_dif_1, masked_loss_1, loss_adversarial_1, consistency_weight))

            writer.add_scalar('Labeled_loss1/loss_seg_dice', loss_seg_dice_1, iter_num)
            writer.add_scalar('Co_loss1/diff_loss', loss_dif_1, iter_num)
            writer.add_scalar('Co_loss1/masked_loss', masked_loss_1, iter_num)
            writer.add_scalar('Co_loss1/adv_loss', loss_adversarial_1, iter_num)
            writer.add_scalar('Co_loss1/consist_weight', consistency_weight, iter_num)

            loss_dif_2 = loss_diff2(y_sig_1_u, y_sig_2_u)
            critic_segs_2 = torch.sigmoid(critic_1(y_sig_1_u))
            masked_loss_2 = loss_mask(y_sig_2_u, y_sig_1_u, critic_segs_2, args.t_m)

            g_critic_segs_2_1 = torch.sigmoid(critic_2(y_all_2))
            g_critic_segs_2_2 = torch.sigmoid(critic_2(labeled_gt_set.float()))
            loss_adversarial_gen_2 = gen_loss(g_critic_segs_2_1)
            loss_adversarial_2 = disc_loss(g_critic_segs_2_1, g_critic_segs_2_2)
            loss_unsup_2 = loss_dif_2 + masked_loss_2 + args.mu * loss_adversarial_gen_2
            loss_2 = args.lamda * loss_sup_2 + consistency_weight * loss_unsup_2

            optimizer_2.zero_grad()
            loss_2.backward()
            optimizer_2.step()

            logging.info(
                'M2 iteration %d : loss : %03f, loss_sup: %03f, loss_diff: %03f, loss_masked: %03f, loss_adv: %03f' % (
                    iter_num, loss_2, loss_sup_2, loss_dif_2, masked_loss_2, loss_adversarial_2))

            writer.add_scalar('Labeled_loss2/loss_seg_dice', loss_seg_dice_2, iter_num)
            writer.add_scalar('Co_loss2/diff_loss', loss_dif_2, iter_num)
            writer.add_scalar('Co_loss2/masked_loss', masked_loss_2, iter_num)
            writer.add_scalar('Co_loss2/adv_loss', loss_adversarial_2, iter_num)

            del loss_1, loss_2, loss_sup_1, loss_sup_2, loss_unsup_1, loss_unsup_2, masked_loss_1, masked_loss_2, loss_dif_1, loss_dif_2, loss_seg_dice_1, loss_seg_dice_2
            gc.collect()
            torch.cuda.empty_cache()

            del g_critic_segs_1_1, g_critic_segs_2_1, g_critic_segs_1_2, g_critic_segs_2_2, y_prob_1, y_all_1, y_sig_1_u, y_prob_2, y_all_2, y_sig_2_u
            gc.collect()
            torch.cuda.empty_cache()

            # Train Discriminator 1
            loss_adversarial_1 = loss_adversarial_1.clone().detach().requires_grad_(True)
            loss_adversarial_2 = loss_adversarial_2.clone().detach().requires_grad_(True)

            dis_optimizer_1.zero_grad()

            critic_loss_1 = loss_adversarial_1

            writer.add_scalar('loss/loss_critic1', critic_loss_1, iter_num)
            critic_loss_1.backward()
            dis_optimizer_1.step()

            torch.cuda.empty_cache()

            # Train Discriminator 2
            dis_optimizer_2.zero_grad()

            critic_loss_2 = loss_adversarial_2

            writer.add_scalar('loss/loss_critic2', critic_loss_2, iter_num)
            critic_loss_2.backward()
            dis_optimizer_2.step()

            torch.cuda.empty_cache()

            if scheduler_1 is not None:
                scheduler_1.step()
            if scheduler_2 is not None:
                scheduler_2.step()

            if c_scheduler_1 is not None:
                c_scheduler_1.step()
            if c_scheduler_2 is not None:
                c_scheduler_2.step()

            if iter_num >= 1000 and iter_num % 100 == 0:
                model_1.eval()

                dice_sample_1 = test_patch_sup.var_all_case(model_1, valloader, patch_size=patch_size, save_result=False, test_save_path=None)

                if dice_sample_1 > best_dice_1:
                    best_dice_1 = dice_sample_1
                    save_best_path = os.path.join(snapshot_path, 'best_model_1.pth'.format(args.model))
                    torch.save(model_1.state_dict(), save_best_path)
                    save_best_pathc = os.path.join(snapshot_path, 'best_critic_1.pth'.format(args.model))
                    torch.save(critic_1.state_dict(), save_best_pathc)
                    logging.info("save best model to {}".format(save_best_path))
                writer.add_scalar('Var_dice1/Dice', dice_sample_1, iter_num)
                writer.add_scalar('Var_dice1/Best_dice', best_dice_1, iter_num)
                logging.info('M1 Best Dice :  %03f Current Dice  %03f' % (best_dice_1, dice_sample_1))
                model_1.train()

                model_2.eval()

                dice_sample_2 = test_patch_sup.var_all_case(model_2, valloader, patch_size=patch_size, save_result=False, test_save_path=None)

                if dice_sample_2 > best_dice_2:
                    best_dice_2 = dice_sample_2
                    save_best_path = os.path.join(snapshot_path, 'best_model_2.pth'.format(args.model))
                    torch.save(model_2.state_dict(), save_best_path)

                    save_best_pathc = os.path.join(snapshot_path, 'best_critic_2.pth'.format(args.model))
                    torch.save(critic_2.state_dict(), save_best_pathc)
                    logging.info("save best model to {}".format(save_best_path))
                writer.add_scalar('Var_dice2/Dice', dice_sample_2, iter_num)
                writer.add_scalar('Var_dice2/Best_dice', best_dice_2, iter_num)
                logging.info('M2 Best Dice :  %03f Current Dice  %03f' % (best_dice_2, dice_sample_2))
                model_2.train()

            if iter_num >= max_iterations:
                save_mode_path_1 = os.path.join(snapshot_path, 'm1_iter_' + str(iter_num) + '.pth')
                torch.save(model_1.state_dict(), save_mode_path_1)
                logging.info("save model 1 to {}".format(save_mode_path_1))

                save_mode_path_2 = os.path.join(snapshot_path, 'm2_iter_' + str(iter_num) + '.pth')
                torch.save(model_2.state_dict(), save_mode_path_2)
                logging.info("save model 2 to {}".format(save_mode_path_2))
                break

        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()