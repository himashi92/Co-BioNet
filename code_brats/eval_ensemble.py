import os
import argparse
import torch

from networks.net_factory import net_factory
from utils.test_patch_ensemble_mc_and import test_all_case
from dataloaders.msd_brats import BraTS
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_name', type=str, default='MSD_BRATS', help='dataset_name')
parser.add_argument('--root_path', type=str, default='/data/himashi/Co-BioNet_MC/', help='Name of Dataset')
parser.add_argument('--exp', type=str, default='Co_BioNet_SEMI', help='exp_name')
parser.add_argument('--model', type=str, default='vnet', help='model_name')
parser.add_argument('--labelnum', type=int, default=39, help='trained samples')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--nms', type=int, default=0, help='apply NMS post-procssing?')
parser.add_argument('--detail', type=int,  default=0, help='print metrics for every samples?')


FLAGS = parser.parse_args()
snapshot_path = FLAGS.root_path + "model/{}_{}_{}_labeled/{}".format(FLAGS.dataset_name, FLAGS.exp, FLAGS.labelnum,
                                                                     FLAGS.model)
test_save_path = FLAGS.root_path + "model/{}_{}_{}_labeled/{}_predictions_ensemble_and/".format(FLAGS.dataset_name, FLAGS.exp,
                                                                                     FLAGS.labelnum, FLAGS.model)

num_classes = 4
patch_size = (128, 128, 96)
FLAGS.root_path = '/data/himashi/Co-BioNet_MC/data/MSD_BRATS/'
FLAGS.max_samples = 387
train_data_path = FLAGS.root_path

db_test = BraTS(base_dir=train_data_path, split='test', patch_size=patch_size)
testloader = DataLoader(db_test, batch_size=1, num_workers=2, pin_memory=True, shuffle=False)

if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)


def calculate_metric():
    net1 = net_factory(net_type='vnet', in_chns=4, class_num=num_classes - 1, mode="test")
    save_mode_path = os.path.join(snapshot_path, 'best_model_1.pth')
    net1.load_state_dict(torch.load(save_mode_path), strict=False)
    print("init weight from {}".format(save_mode_path))
    net1.eval()

    net2 = net_factory(net_type='vnet', in_chns=4, class_num=num_classes - 1, mode="test")
    save_mode_path2 = os.path.join(snapshot_path, 'best_model_2.pth')
    net2.load_state_dict(torch.load(save_mode_path2), strict=False)
    print("init weight from {}".format(save_mode_path2))
    net2.eval()

    avg_metric = test_all_case(net1, net2, testloader, patch_size=patch_size, save_result=True, test_save_path=test_save_path)

    return avg_metric


if __name__ == '__main__':
    metric = calculate_metric()
    print(metric)
