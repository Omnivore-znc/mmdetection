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
#cd /jayden/mmdetection_merge
#export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64
#cd ..
#python setup.py develop
#chmod 777 ./tools/ -R
export CUDA_VISIBLE_DEVICES=0,3,4,5
./tools/dist_train2.sh ./configs_znc/blaze_body_keypoint2.py 4 --validate