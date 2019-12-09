import timeit
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import mmcv
import torch
import sys
sys.path.append('../')
from mmdetection.mmdet.apis import inference

img_types = ['.jpg','.jpeg','.png','.bmp']
class_names = ['toujian']

def runit0(model_config, weights, image_dir, out_dir):
    model = inference.init_detector(model_config,weights)
    walker = os.walk(image_dir);
    run_num = 10000
    for dir_path, sub_dirs, file_names in walker:
        for filename in file_names:
            ext = os.path.splitext(filename)[-1]
            if ext in img_types:
                img = os.path.join(dir_path,filename)
                start = timeit.default_timer()
                results = inference.inference_detector(model,img)
                end = timeit.default_timer()
                print('{}'.format((end-start)*1000))
                out_file = os.path.join(out_dir,filename)
                inference.show_result(img,result=results,class_names=class_names,show = False,out_file=out_file)
                run_num=run_num-1
                print('{}: {}'.format(run_num,filename))
                if run_num<1:
                    return

def runit1(model_config, weights, image_dir, image_list, out_dir):
    model = inference.init_detector(model_config,weights)
    with open(image_list) as list_file:
        run_num = 100
        all_filenames = list_file.readlines()
        for filename in all_filenames:
            filename = filename.strip()
            img = os.path.join(image_dir,filename+'.jpg')
            start = timeit.default_timer()
            results = inference.inference_detector(model,img)
            end = timeit.default_timer()
            print('{}'.format((end - start) * 1000))
            out_file = os.path.join(out_dir,filename+'.jpg')
            inference.show_result(img,result=results,class_names=class_names,show = False,out_file=out_file)
            run_num=run_num-1
            print('{}: {}'.format(run_num,filename))
            if run_num<1:
                return

if __name__=='__main__':
    model_config = '../configs/ga_faster_r50_caffe_fpn_1x_head.py'
    model_weight = '../../mmdet_models/work_dirs/ga_faster_rcnn_r50_caffe_fpn_1x/epoch_12.pth'
    out_dir = '../../mmdet_models/work_dirs/ga_faster_rcnn_r50_caffe_fpn_1x'

    img_dir0 = '/opt/space_host/data_xiaozu/head_data/application-test/rentou-test-set'
    runit0(model_config, model_weight, img_dir0, out_dir)

    image_dir1 = '/opt/space_host/data_xiaozu/head_data/CrowdHuman/JPEGImages'
    image_list = '/opt/space_host/data_xiaozu/head_data/CrowdHuman/ImageSets/Main/test.txt'
    #runit1(model_config, model_weight, image_dir1, image_list, out_dir)


