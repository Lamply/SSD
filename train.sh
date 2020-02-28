#!/bin/sh

retry_num=0
while [ ! -e /home/lamply/projects/SSD/outputs/efficient_net_b3_ssd300_voc0712/model_final.pth ]
do
  echo "retry: "
  echo $retry_num
  retry_num=$((retry_num+1))
  if [ "$retry_num" -gt 100000 ]; then
    touch /home/lamply/projects/SSD/outputs/efficient_net_b3_ssd300_voc0712/model_final.pth
  fi
  python train.py --config-file configs/efficient_net_b3_ssd300_voc0712.yaml 2>&1 | tee -a terminal_2.log
done

echo "training complete."
