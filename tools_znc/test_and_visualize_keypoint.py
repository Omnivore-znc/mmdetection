import timeit
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import cv2
import sys
sys.path.append('../')
from mmdetection.mmdet.apis import inference

img_types = ['.jpg','.jpeg','.png','.bmp']
class_names = ['hidden','visible']

def visualize_results(points, img_path, save_path):
    points_tmp = points.cpu()
    img = cv2.imread(img_path)
    img = cv2.resize(img, (64, 128))
    for i in range(points_tmp.size()[0]):
        if points_tmp[i,2]==1:
            cv2.circle(img, (int(points_tmp[i,0]), int(points_tmp[i,1])), 2, (0, 0, 255), thickness=2)
        elif points_tmp[i,2]==2:
            cv2.circle(img,(int(points_tmp[i,0]), int(points_tmp[i,1])),2,(255,0,0),thickness=2)
    img = cv2.resize(img, (0, 0), fx=2.0, fy=2.0)
    #dir_name,file_name = os.path.split(img_path)
    #dst_path = os.path.join(save_path,file_name)
    cv2.imwrite(save_path,img)

def runit0(model_config, weights, image_dir, out_dir):
    model = inference.init_detector(model_config, weights)
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
                #inference.show_result(img,result=results,class_names=class_names,show = False,out_file=out_file)
                visualize_results(results[0], img, out_file)
                run_num=run_num-1
                print('{}: {}'.format(run_num,filename))
                if run_num<1:
                    return

if __name__=='__main__':
    model_config = '../configs_znc/blaze_body_keypoint.py'
    model_weight = '/opt/space_host/zhongnanchang/mmdet_models/work_dirs/blaze_body_keypoint/epoch_100.pth'
    out_dir = '/opt/space_host/zhongnanchang/mmdet_models/work_dirs/blaze_body_keypoint'

    img_dir0 = '/opt/space_host/data_xiaozu/keypoint_coco2017/self-test-set'
    runit0(model_config, model_weight, img_dir0, out_dir)


