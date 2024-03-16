#!/bin/bash


 for dataset in 'mnist' 'fashion-mnist'
 do
     for model in 'resnet50' 'resnext50' 'wide_resnet50' 'alexnet' 'vgg16' 'senet34' 'densenet121' \
                 'simplenetv1' 'efficientnetv2s' 'googlenet' 'xception' 'mobilenetv2' 'inceptionv3' \
                 'shufflenetv2' 'squeezenet' 'regnet_y_400mf'
     do
         python3 train.py \
             --data_name $dataset \
             --model_name $model \
             --mu_threshold 0.5 \
             --load_baseline \
             --baseline_model_path "./work_dir/baseline/"${dataset}_${model}"/best-model.pth" \
             --lr 0.01 \
             --epochs 100 \
             --gpu_id 0 \
             --print_freq 100 \
             > $(date "+%Y%m%d_%H%M%S")_${dataset}_${model}.log
    
         wait
     done
 done

