import itertools
import sys
import mmcv
import numpy as np
import copy

from terminaltables import AsciiTable
from time import time, clock
from .recall import eval_recalls
from tqdm import tqdm


def keypoints_eval(result_files,
              result_types,
              dataset,
              # coco,
              max_dets=(100, 300, 1000),
              classwise=False):

    for res_type in result_types:
        assert res_type in [
            'human-points'
        ]

    eval_type = 'oks'  # pck or oks

    det_results = mmcv.load(result_files)

    # num_img x num_obj_per_img x 17 x 3
    # print(len(det_results), len(det_results[0]), det_results[0][0].shape)
    # print(det_results)


    gt_points = []
    gt_labels = []
    gt_normalize = []
    gt_height = []
    gt_width = []
    # print(len(dataset))

    num_points = 0

    for i in tqdm(range(len(dataset))):
        # print(i)
        img = dataset.load_annotations
        ann = dataset.get_ann_info(i)
        points = ann['points']
        labels = ann['labels']
        gt_points.append(points)
        gt_labels.append(labels)

        for v in labels:
            if v==1 or v==2:
                num_points+=1


        if eval_type == 'pck':
            normalize = np.sqrt(ann['height']**2+ann['width']**2)
            gt_normalize.append(normalize)
        elif eval_type == 'oks':
            gt_height.append(ann['height'])
            gt_width.append(ann['width'])

        # print(type(points))
        # print(ann)

    print('num points:', num_points)
    if eval_type=='pck':
        pck_eval(det_results, gt_points, gt_labels, gt_normalize)
    elif eval_type=='oks':
        computeOks(det_results, gt_points, gt_labels, gt_height, gt_width)


def pck_eval(preds, gt_points, gt_labels, gt_normalize, bound=0.05):
    """
    Use PCK with threshold of .5 of normalized distance (presumably head size)

    normalizing: 每张图的 num_train??

    coco:
    "nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder",
    "left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee", "left_ankle","right_ankle"

    """

    assert len(preds)==len(gt_points)==len(gt_labels)

    correct = {'all': {'total': 0,
                       'ankle': 0, 'knee': 0, 'hip': 0, 'ear': 0, 'eye': 0, 'nose': 0,
                       'wrist': 0, 'elbow': 0, 'shoulder': 0},
               'visible': {'total': 0,
                       'ankle': 0, 'knee': 0, 'hip': 0, 'ear': 0, 'eye': 0, 'nose': 0,
                       'wrist': 0, 'elbow': 0, 'shoulder': 0},
               'not_visible': {'total': 0,
                       'ankle': 0, 'knee': 0, 'hip': 0, 'ear': 0, 'eye': 0, 'nose': 0,
                       'wrist': 0, 'elbow': 0, 'shoulder': 0},
               }

    joint_list = ["nose","eye","eye","ear","ear",
                  "shoulder","shoulder","elbow","elbow","wrist","wrist",
                  "hip","hip","knee","knee","ankle","ankle"]

    assert len(joint_list)==17

    count = copy.deepcopy(correct)

    for pred, gpoint, vlabel, normalize in tqdm(zip(preds, gt_points, gt_labels, gt_normalize)):

        pred_np = pred[0].cpu().numpy()

        # normalize = 0

        # compute normalize by torse size
        # if (vlabel[5]==0 or vlabel[12]==0) and (vlabel[6]==0 or vlabel[11]==0):
        #     continue
        #
        # elif vlabel[5]==1 and vlabel[12]==1:
        #     normalize = np.linalg.norm(gpoint[5, :2] - gpoint[12, :2])
        #
        # else:
        #     normalize = np.linalg.norm(gpoint[6, :2] - gpoint[11, :2])


        # 每一张图每一个点

        for j in range(gpoint.shape[0]):
            vis = 'visible'
            if vlabel[j] == 0:  ## not in picture!
                continue
            if vlabel[j] == 1:
                vis = 'not_visible'
            joint = joint_list[j]

            # if idx >= num_train:
            count['all']['total'] += 1
            count['all'][joint] += 1
            count[vis]['total'] += 1
            count[vis][joint] += 1

            # 计算2范数
            # normalize表示躯干距离？？

            error = np.linalg.norm(pred_np[j, :2] - gpoint[j, :2]) / normalize

            if bound > error:
                correct['all']['total'] += 1
                correct['all'][joint] += 1
                correct[vis]['total'] += 1
                correct[vis][joint] += 1

    ## breakdown by validation set / training set
    for k in correct:
        print(k, ':')
        for key in correct[k]:
            print('Val PCK @', bound, ',', key, ':', round(correct[k][key] / max(count[k][key], 1), 3), ', count:', count[k][key])

        print('\n')

def computeOks(preds, gt_points, gt_labels, gt_height, gt_width):

    '''

    :param preds:  Tensor: nx1x17x3
    :param gt_points: numpy: nx17x2
    :param gt_labels: nx17
    :return:
    '''

    # setKpParams
    # 已经根据gt算出来了？？
    kpt_oks_sigmas = np.array(
        [.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0

    sigmas = kpt_oks_sigmas
    vars = (sigmas * 2) ** 2
    k = len(sigmas) # 17

    oks_list = []

    for pred, gpoint, vlabel, height, width in tqdm(zip(preds, gt_points, gt_labels, gt_height, gt_width)):

        # compute oks between each detection and ground truth object
        g = np.array(gpoint)
        xg = g[:, 0]
        yg = g[:, 1]
        vg = vlabel
        k1 = np.count_nonzero(vg > 0)

        x0 = 0
        x1 = width
        y0 = 0
        y1 = height

        gt_area = width*height

        # 17x3
        d = pred[0].cpu().numpy()
        xd = d[:, 0]
        yd = d[:, 1]
        if k1 > 0:
            # measure the per-keypoint distance if keypoints visible
            dx = xd - xg
            dy = yd - yg
        else:
            # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
            z = np.zeros((k))
            dx = np.max((z, x0 - xd), axis=0) + np.max((z, xd - x1), axis=0)
            dy = np.max((z, y0 - yd), axis=0) + np.max((z, yd - y1), axis=0)
        e = (dx ** 2 + dy ** 2) / vars / (gt_area + np.spacing(1)) / 2
        if k1 > 0:
            e = e[vg > 0]
        oks = np.sum(np.exp(-e)) / e.shape[0]
        oks_list.append(oks)

    print("Mean OKS = {}".format(np.mean(oks_list)))

    oks_list_np = np.array(oks_list)
    bound_list = np.arange(0.5, 1.0, 0.05)

    # print(oks_list_np.shape, bound_list.shape)

    mAP = 0.0

    for bound in bound_list:
        num_oks_bound = np.count_nonzero(oks_list_np>bound)
        AP = num_oks_bound / oks_list_np.shape[0]
        print('Val AP @ {:.2f} : {:.3f}'.format(bound, AP))

        mAP += AP

    print('Val mAP : {:.3f}'.format(mAP/bound_list.shape[0]))

    print('\n')

    return oks_list

# hourglasstensorflow
class PCKEval():

    def __init__(self):
        pass


    def pcki(self, joint_id, gtJ, prJ, idlh=3, idrs=12):
        """ Compute PCK accuracy on a given joint
        Args:
            joint_id	: Index of the joint considered
            gtJ			: Ground Truth Joint
            prJ			: Predicted Joint
            idlh		: Index of Normalizer (Left Hip on PCK, neck on PCKh)
            idrs		: Index of Normalizer (Right Shoulder on PCK, top head on PCKh)
        Returns:
            (float) NORMALIZED L2 ERROR
        """
        return np.linalg.norm(gtJ[joint_id] - prJ[joint_id][::-1]) / np.linalg.norm(gtJ[idlh] - gtJ[idrs])


    def pck(self, weight, gtJ, prJ, gtJFull, boxL, idlh=3, idrs=12):
        """ Compute PCK accuracy for a sample
        Args:
            weight		: Index of the joint considered
            gtJFull	: Ground Truth (sampled on whole image)
            gtJ			: Ground Truth (sampled on reduced image)
            prJ			: Prediction
            boxL		: Box Lenght
            idlh		: Index of Normalizer (Left Hip on PCK, neck on PCKh)
            idrs		: Index of Normalizer (Right Shoulder on PCK, top head on PCKh)
        """
        for i in range(len(weight)):
            if weight[i] == 1:
                self.ratio_pck.append(self.pcki(i, gtJ, prJ, idlh=idlh, idrs=idrs))
                self.ratio_pck_full.append(self.pcki(i, gtJFull, np.asarray(prJ / 255 * boxL)))
                self.pck_id.append(i)


    def compute_pck(self, datagen, idlh=3, idrs=12, testSet=None):
        """ Compute PCK on dataset
        Args:
            datagen	: (DataGenerator)
            idlh		: Index of Normalizer (Left Hip on PCK, neck on PCKh)
            idrs		: Index of Normalizer (Right Shoulder on PCK, top head on PCKh)
        """
        datagen.pck_ready(idlh=idlh, idrs=idrs, testSet=testSet)
        self.ratio_pck = []
        self.ratio_pck_full = []
        self.pck_id = []
        samples = len(datagen.pck_samples)
        startT = time()
        for idx, sample in enumerate(datagen.pck_samples):
            percent = ((idx + 1) / samples) * 100
            num = np.int(20 * percent / 100)
            tToEpoch = int((time() - startT) * (100 - percent) / (percent))
            sys.stdout.write('\r PCK : {0}>'.format("=" * num) + "{0}>".format(" " * (20 - num)) + '||' + str(percent)[
                                                                                                          :4] + '%' + ' -timeToEnd: ' + str(
                tToEpoch) + ' sec.')
            sys.stdout.flush()
            res = datagen.getSample(sample)
            if res != False:
                img, gtJoints, w, gtJFull, boxL = res
                prJoints = self.joints_pred_numpy(np.expand_dims(img / 255, axis=0), coord='img', thresh=0)
                self.pck(w, gtJoints, prJoints, gtJFull, boxL, idlh=idlh, idrs=idrs)
        print('Done in ', int(time() - startT), 'sec.')





