import timeit
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import cv2
import sys
sys.path.append('../')
from mmdet.apis import inference

img_types = ['.jpg','.jpeg','.png','.bmp']
class_names = ['hidden','visible']

def draw_line(img, points, idx0, idx1):
    idx0 = idx0 - 1
    idx1 = idx1 - 1
    if points[idx0, 2] != 0 and points[idx1, 2] != 0:
        cv2.line(img,
                 (int(points[idx0, 0]), int(points[idx0, 1])),
                 (int(points[idx1, 0]), int(points[idx1, 1])),
                 (0, 255, 255),
                 2)
    # cv2.line(img,
    #          (int(points[idx0, 0]), int(points[idx0, 1])),
    #          (int(points[idx1, 0]), int(points[idx1, 1])),
    #          (0, 255, 255),
    #          2)

def visualize_results(points, img_path, save_path):
    points_tmp = points.cpu()
    img = cv2.imread(img_path)
    #img = cv2.resize(img, (64, 128))
    enlarge = 2.0
    img = cv2.resize(img, (0, 0), fx=enlarge, fy=enlarge)
    points_tmp[:,:2] = points_tmp[:,:2]*enlarge

    #left_ankle - left_knee
    idx_0,idx_1 = 16, 14
    draw_line(img, points_tmp,idx_0,idx_1)
    #left_knee - left_hip
    idx_0, idx_1 = 14, 12
    draw_line(img, points_tmp, idx_0, idx_1)
    #right_ankle - right_knee
    idx_0, idx_1 = 17, 15
    draw_line(img, points_tmp, idx_0, idx_1)
    # right_knee - right_hip
    idx_0, idx_1 = 15, 13
    draw_line(img, points_tmp, idx_0, idx_1)
    #left_hip - right_hip
    idx_0, idx_1 = 12, 13
    draw_line(img, points_tmp, idx_0, idx_1)
    #left_shoulder - left_hip
    idx_0, idx_1 = 6, 12
    draw_line(img, points_tmp, idx_0, idx_1)
    #right_shoulder - right_hip
    idx_0, idx_1 = 7, 13
    draw_line(img, points_tmp, idx_0, idx_1)
    #left_shoulder - right_shoulder
    idx_0, idx_1 = 6, 7
    draw_line(img, points_tmp, idx_0, idx_1)
    #left_shoulder - left_elbow
    idx_0, idx_1 = 6, 8
    draw_line(img, points_tmp, idx_0, idx_1)
    #right_shoulder - right_elbow
    idx_0, idx_1 = 7, 9
    draw_line(img, points_tmp, idx_0, idx_1)
    #left_elbow - left_wrist
    idx_0, idx_1 = 8, 10
    draw_line(img, points_tmp, idx_0, idx_1)
    #right_elbow - right_wrist
    idx_0, idx_1 = 9, 11
    draw_line(img, points_tmp, idx_0, idx_1)
    #left_eye - right_eye
    idx_0, idx_1 = 2, 3
    draw_line(img, points_tmp, idx_0, idx_1)
    #nose - left_eye
    idx_0, idx_1 = 1, 2
    draw_line(img, points_tmp, idx_0, idx_1)
    #nose - right_eye
    idx_0, idx_1 = 1, 3
    draw_line(img, points_tmp, idx_0, idx_1)
    #left_eye - left_ear
    idx_0, idx_1 = 2, 4
    draw_line(img, points_tmp, idx_0, idx_1)
    #right_eye - right_ear
    idx_0, idx_1 = 3, 5
    draw_line(img, points_tmp, idx_0, idx_1)
    idx_0, idx_1 = 4, 6
    draw_line(img, points_tmp, idx_0, idx_1)
    idx_0, idx_1 = 5, 7
    draw_line(img, points_tmp, idx_0, idx_1)

    ###
    for i in range(points_tmp.size()[0]):
        if points_tmp[i, 2] == 1 and i % 2 == 1:
            cv2.circle(img, (int(points_tmp[i, 0]), int(points_tmp[i, 1])), 3, (0, 0, 255), thickness=2)
        elif points_tmp[i, 2] == 1 and i % 2 == 0:
            cv2.rectangle(img, (int(points_tmp[i, 0]), int(points_tmp[i, 1])),
                          (int(points_tmp[i, 0]) + 10, int(points_tmp[i, 1]) + 10), (0, 0, 255))
        elif points_tmp[i, 2] == 2 and i % 2 == 1:
            cv2.circle(img, (int(points_tmp[i, 0]), int(points_tmp[i, 1])), 3, (255, 0, 0), thickness=2)
        elif points_tmp[i, 2] == 2 and i % 2 == 0:
            cv2.rectangle(img, (int(points_tmp[i, 0]), int(points_tmp[i, 1])),
                          (int(points_tmp[i, 0]) + 10, int(points_tmp[i, 1]) + 10), (255, 0, 0))
        elif points_tmp[i, 2] == 0:
            cv2.circle(img, (int(points_tmp[i, 0]), int(points_tmp[i, 1])), 3, (0, 255, 0), thickness=2)
        else:
            cv2.circle(img, (int(points_tmp[i, 0]), int(points_tmp[i, 1])), 3, (0, 255, 255), thickness=2)

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
                if run_num%10==0:
                    print('run_num = '.format(run_num))
def runit1(model_config, weights, image_list, out_dir):
    model = inference.init_detector(model_config, weights)
    img_dir,_ = os.path.split(image_list)
    with open(image_list) as img_file:
        linestr = img_file.readline().strip()
        run_num = 1000
        while len(linestr)>2 and run_num>0:
            img_path = os.path.join(img_dir,'IMAGE_ANNOTATIONS/'+linestr+'.jpg')
            if not os.path.exists(img_path):
                continue
            start = timeit.default_timer()
            results = inference.inference_detector(model, img_path)
            end = timeit.default_timer()
            print('{}'.format((end - start) * 1000))
            out_file = os.path.join(out_dir, linestr+'.jpg')
            # inference.show_result(img,result=results,class_names=class_names,show = False,out_file=out_file)
            visualize_results(results[0], img_path, out_file)
            run_num = run_num - 1
            print('{}: {}'.format(run_num, linestr+'.jpg'))
            if run_num < 1:
                return
            linestr = img_file.readline().strip()
            run_num -= 1

if __name__=='__main__':
    model_config = '/opt/space_host/zhongnanchang/mmdet_models/work_dirs/resnet_body_keypoint1912281100/resnet_body_keypoint.py'
    model_weight = '/opt/space_host/zhongnanchang/mmdet_models/work_dirs/resnet_body_keypoint1912281100/epoch_300.pth'
    out_dir = '/opt/space_host/zhongnanchang/mmdet_models/work_dirs/resnet_body_keypoint1912281100/results'

    img_dir0 = '/opt/space_host/data_xiaozu/keypoint_coco2017/self-test-set_from_reid'
    runit0(model_config, model_weight, img_dir0, out_dir)

    img_dir0 = '/opt/space_host/data_xiaozu/person_body'
    runit0(model_config, model_weight, img_dir0, out_dir)

    img_list = '/opt/space_host/data_xiaozu/keypoint_coco2017/idx_list-21w_train.txt'
    runit1(model_config, model_weight, img_list, out_dir)


