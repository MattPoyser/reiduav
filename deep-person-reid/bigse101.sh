#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
##SBATCH -p gpu-private
#SBATCH -p res-gpu-small
#SBATCH --qos=short
#SBATCH --job-name=bigse101
#SBATCH --gres=gpu:1
#SBATCH -o bigse101.out
#SBATCH --mem=24g
#SBATCH -t 2-0:0:0

source ../venv/bin/activate

#pip3 install -r deep-person-reid/requirements.txt
#python3 setup.py develop
python3 scripts/MarsBest.py --config-file configs/bigtosmalllong.yaml --transforms random_flip random_erase --root /home2/lgfm95/reid/uavdata/ --save_path "bigse101.pth" --ncc True --epochs 500 --model_name se_resnet101 --model_path none > bigse101.1.out 2> bigse101.1.err
python3 scripts/MarsBest.py --config-file configs/bigtosmalllong.yaml --transforms random_flip random_erase --root /home2/lgfm95/reid/uavdata/ --save_path "bigse101.pth" --ncc True --epochs 500 --model_name se_resnet101 --model_path none > bigse101.2.out 2> bigse101.2.err
python3 scripts/MarsBest.py --config-file configs/bigtosmalllong.yaml --transforms random_flip random_erase --root /home2/lgfm95/reid/uavdata/ --save_path "bigse101.pth" --ncc True --epochs 500 --model_name se_resnet101 --model_path none > bigse101.3.out 2> bigse101.3.err
