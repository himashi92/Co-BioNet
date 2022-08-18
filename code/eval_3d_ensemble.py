import os
import argparse
import torch

from networks.net_factory import net_factory
from utils.test_patch_ensemble import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='LA', help='dataset_name')
parser.add_argument('--root_path', type=str, default='./', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='Co_BioNet', help='exp_name')
parser.add_argument('--model', type=str, default='vnet', help='model_name')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--detail', type=int, default=1, help='print metrics for every samples?')
parser.add_argument('--labelnum', type=int, default=8, help='labeled data')
parser.add_argument('--nms', type=int, default=0, help='apply NMS post-procssing?')

FLAGS = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path ="./model/{}_{}_{}_labeled/{}".format(FLAGS.dataset_name, FLAGS.exp, FLAGS.labelnum,
                                                                     FLAGS.model)
test_save_path = "./model/{}_{}_{}_labeled/{}_predictions_ensemble/".format(FLAGS.dataset_name, FLAGS.exp,
                                                                                   FLAGS.labelnum, FLAGS.model)

num_classes = 2
if FLAGS.dataset_name == "LA":
    patch_size = (112, 112, 80)
    FLAGS.root_path = '../data/LA'
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = [FLAGS.root_path + "/2018LA_Seg_Training Set/" + item.replace('\n', '') + "/mri_norm2.h5" for item in
                  image_list]

elif FLAGS.dataset_name == "Pancreas_CT":
    patch_size = (96, 96, 96)
    FLAGS.root_path = '../data/Pancreas'
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = [FLAGS.root_path + "/Pancreas_h5/" + item.replace('\n', '') + "_norm.h5" for item in image_list]

if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)
print(test_save_path)


def test_calculate_metric():
    net_1 = net_factory(net_type='vnet', in_chns=1, class_num=num_classes - 1, mode="test")
    save_mode_path_1 = os.path.join(snapshot_path, 'best_model_1.pth'.format(FLAGS.model))
    net_1.load_state_dict(torch.load(save_mode_path_1), strict=False)
    print("init weight from {}".format(save_mode_path_1))
    net_1.eval()

    net_2 = net_factory(net_type='vnet', in_chns=1, class_num=num_classes - 1, mode="test")
    save_mode_path_2 = os.path.join(snapshot_path, 'best_model_2.pth'.format(FLAGS.model))
    net_2.load_state_dict(torch.load(save_mode_path_2), strict=False)
    print("init weight from {}".format(save_mode_path_2))
    net_2.eval()

    if FLAGS.dataset_name == "LA":
        avg_metric = test_all_case(FLAGS.model, 1, net_1, net_2, image_list, num_classes=num_classes,
                                   patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                                   save_result=True, test_save_path=test_save_path,
                                   metric_detail=FLAGS.detail, nms=FLAGS.nms)
    elif FLAGS.dataset_name == "Pancreas_CT":
        avg_metric = test_all_case(FLAGS.model, 1, net_1, net_2, image_list, num_classes=num_classes,
                                   patch_size=(96, 96, 96), stride_xy=16, stride_z=16,
                                   save_result=True, test_save_path=test_save_path,
                                   metric_detail=FLAGS.detail, nms=FLAGS.nms)

    return avg_metric


if __name__ == '__main__':
    metric = test_calculate_metric()
    print(metric)
