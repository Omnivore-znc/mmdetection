
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torch
num_classes = 11
# model_coco = torch.load("/ssd/yckj1758/pretrained_models/cascade_rcnn_x101_64x4d_fpn_2x_20181218-5add321e.pth")
model_coco = torch.load("/ssd/yckj1758/pretrained_models/faster_rcnn_x101_64x4d_fpn_1x.pth")

# weight
for a in list(model_coco["state_dict"]):

    if a.startswith('backbone'):
        c = a.replace('backbone.',"")
        model_coco["state_dict"][c] = model_coco["state_dict"][a]
    del model_coco["state_dict"][a]

# weight
# model_coco["state_dict"]["bbox_head.fc_cls.weight"] = model_coco["state_dict"]["bbox_head.fc_cls.weight"][:num_classes, :]
# model_coco["state_dict"]["bbox_head.0.fc_cls.weight"] = model_coco["state_dict"]["bbox_head.0.fc_cls.weight"][:num_classes, :]
# model_coco["state_dict"]["bbox_head.1.fc_cls.weight"] = model_coco["state_dict"]["bbox_head.1.fc_cls.weight"][:num_classes, :]
# model_coco["state_dict"]["bbox_head.2.fc_cls.weight"] = model_coco["state_dict"]["bbox_head.2.fc_cls.weight"][:num_classes, :]
# # # bias
# # model_coco["state_dict"]["bbox_head.fc_cls.weight"] = model_coco["state_dict"]["bbox_head.fc_cls.weight"][:num_classes, :]
# model_coco["state_dict"]["bbox_head.0.fc_cls.bias"] = model_coco["state_dict"]["bbox_head.0.fc_cls.bias"][:num_classes]
# model_coco["state_dict"]["bbox_head.1.fc_cls.bias"] = model_coco["state_dict"]["bbox_head.1.fc_cls.bias"][:num_classes]
# model_coco["state_dict"]["bbox_head.2.fc_cls.bias"] = model_coco["state_dict"]["bbox_head.2.fc_cls.bias"][:num_classes]

# save new model
torch.save(model_coco, "/ssd/yckj1758/pretrained_models/sub4_fpn_pretrained_weights_classes_%d.pth" % num_classes)