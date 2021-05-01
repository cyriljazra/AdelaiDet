#!/bin/bash
OUTPUT_DIR=/home/cyriljazra/AdelaiDet/training_dir/SOLOv2_R50_3x
export DETECTRON2_DATASETS=/home/cyriljazra/datasets/
RANK=$1
DIST_URL=$2
PORT=7869
echo "RANK: $RANK"
echo "DIST URL: $DIST_URL"
echo "CURRENT HOST: $HOSTNAME"
/home/cyriljazra/.conda/envs/opence_clone/bin/python /home/cyriljazra/AdelaiDet/tools/train_net.py --config-file configs/SOLOv2/R50_3x.yaml \
	--num-machines 2 --machine-rank $RANK --dist-url $DIST_URL --num-gpus 4 --resume OUTPUT_DIR $OUTPUT_DIR
