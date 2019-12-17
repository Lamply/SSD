#!/bin/sh

retry_num=0
while [ ! -e /home/lamply/projects/SSD/outputs/mobilenet_v2_ssd320_coco_trainval35k/model_final.pth ]
do
  echo "retry: "
  echo $retry_num
  retry_num=$((retry_num+1))
  if [ "$retry_num" -gt 100000 ]; then
    touch /home/lamply/projects/SSD/outputs/mobilenet_v2_ssd320_coco_trainval35k/model_final.pth
  fi
  python train.py --config-file configs/mobilenet_v2_ssd320_coco_trainval35k.yaml 2>&1 | tee -a terminal.log
done

echo "training complete."
