# #!/bin/bash
# # >>> conda initialize >>>
# # !! Contents within this block are managed by 'conda init' !!
# __conda_setup="$('/root/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
# if [ $? -eq 0 ]; then
    # eval "$__conda_setup"
# else
    # if [ -f "/root/anaconda3/etc/profile.d/conda.sh" ]; then
        # . "/root/anaconda3/etc/profile.d/conda.sh"
    # else
        # export PATH="/root/anaconda3/bin:$PATH"
    # fi
# fi
# unset __conda_setup
# # <<< conda initialize <<<
# conda activate open-mmlab
# #conda activate
# #cd $(dirname $0)
# #echo "$(pwd)"


cd /opt/space_host/zhongnanchang/mmdet/mmdetection
python setup.py develop

chmod 777 ./tools/ -R

#export CUDA_VISIBLE_DEVICES=0,1,2,3,5,6
./tools/dist_train.sh /opt/space_host/zhongnanchang/mmdet/configs/ga_faster_r50_caffe_fpn_1x_head.py 8 --validate