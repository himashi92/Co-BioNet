import h5py
import math
import nibabel as nib
import numpy as np
from medpy import metric
import torch
import torch.nn.functional as F
from tqdm import tqdm
from skimage.measure import label
import torchio as tio


def getLargestCC(segmentation):
    labels = label(segmentation)
    assert (labels.max() != 0)  # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC


def test_all_case(model_name, num_outputs, model_1, model_2, image_list, num_classes, patch_size=(112, 112, 80), stride_xy=18,
                  stride_z=4, save_result=True, test_save_path=None, preproc_fn=None, metric_detail=1, nms=0):
    loader = tqdm(image_list) if not metric_detail else image_list
    ith = 0
    total_metric = 0.0
    total_metric_average = 0.0
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        if preproc_fn is not None:
            image = preproc_fn(image)
        prediction_1, score_map_1, artefact_image = test_single_case_first_output(model_1, image, stride_xy, stride_z, patch_size,
                                                              num_classes=num_classes)
        prediction_2, score_map_2, artefact_image = test_single_case_first_output(model_2, image, stride_xy, stride_z, patch_size,
                                                                  num_classes=num_classes)

        if nms:
            prediction_1 = getLargestCC(prediction_1)
            prediction_2 = getLargestCC(prediction_2)

        prediction = np.logical_or(prediction_1, prediction_2)
        all_single_metric = []
        if np.sum(prediction) == 0:
            single_metric_e = (0, 0, 0, 0)
            all_single_metric.append(single_metric_e[0])
        else:
            single_metric_e = calculate_metric_percase(prediction, label[:])
            all_single_metric.append(single_metric_e[0])

        if np.sum(prediction_1) == 0:
            single_metric_1 = (0, 0, 0, 0)
            all_single_metric.append(single_metric_1[0])
        else:
            single_metric_1 = calculate_metric_percase(prediction_1, label[:])
            all_single_metric.append(single_metric_1[0])

        if np.sum(prediction_2) == 0:
            single_metric_2 = (0, 0, 0, 0)
            all_single_metric.append(single_metric_2)
        else:
            single_metric_2 = calculate_metric_percase(prediction_2, label[:])
            all_single_metric.append(single_metric_2[0])

        max_single_matrix = np.max(all_single_metric)
        result = np.where(all_single_metric == max_single_matrix)
        #print(result[0][0])

        if result[0][0] == 0:
            single_metric = single_metric_e
            best_prediction = prediction
        elif result[0][0] == 1:
            single_metric = single_metric_1
            best_prediction = prediction_1
        else:
            single_metric = single_metric_2
            best_prediction = prediction_2

        if metric_detail:
            print('%02d,\t%.5f, %.5f, %.5f, %.5f' % (
            ith, single_metric[0], single_metric[1], single_metric[2], single_metric[3]))

        total_metric += np.asarray(single_metric)

        if save_result:
            nib.save(nib.Nifti1Image(best_prediction.astype(np.float32), np.eye(4)),
                     test_save_path + "%02d_pred.nii.gz" % ith)
            nib.save(nib.Nifti1Image(artefact_image.astype(np.float32), np.eye(4)), test_save_path + "%02d_img_artefact.nii.gz" % ith)
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), test_save_path + "%02d_gt.nii.gz" % ith)

        ith += 1

    avg_metric = total_metric / len(image_list)
    print('average metric is decoder 1 {}'.format(avg_metric))

    with open(test_save_path + '../{}_performance_best_artefact_motion.txt'.format(model_name), 'w') as f:
        f.writelines('average metric of decoder 1 is {} \n'.format(avg_metric))
    return avg_metric


def test_single_case_first_output(model, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)], mode='constant',
                       constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    artefacted_image = np.zeros((num_classes,) + image.shape).astype(np.float32)
    score_map = np.zeros((num_classes,) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd - patch_size[2])
                test_patch = image[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                np.expand_dims(np.expand_dims(test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch)

                motion = tio.RandomMotion()
                noise = tio.RandomNoise(mean=(0,5), std=(0,5))
                ghost = tio.RandomGhosting()
                artefact = motion
                test_patch = artefact(test_patch.unsqueeze(0)).unsqueeze(0)
                #test_patch = test_patch.unsqueeze(0).unsqueeze(0)

                test_patch = test_patch.cuda()

                with torch.no_grad():
                    y = model(test_patch)

                    y = torch.sigmoid(y)
                y = y.cpu().data.numpy()
                test_patch = test_patch.cpu().data.numpy()

                artefacted_image[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = artefacted_image[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + test_patch
                score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + y
                cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + 1

    artefacted_image = artefacted_image / np.expand_dims(cnt, axis=0)
    score_map = score_map / np.expand_dims(cnt, axis=0)
    label_map = (score_map[0] > 0.5).astype(np.int)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        score_map = score_map[:, wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
    return label_map, score_map, artefacted_image[0]


def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dice, jc, hd, asd
