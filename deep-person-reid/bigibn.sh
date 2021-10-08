#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
##SBATCH -p gpu-private
#SBATCH -p res-gpu-small
#SBATCH --qos=short
#SBATCH --job-name=bigibn
#SBATCH --gres=gpu:1
#SBATCH -o bigibn.out
#SBATCH --mem=24g
#SBATCH -t 2-0:0:0

source ../venv/bin/activate

#pip3 install -r deep-person-reid/requirements.txt
#python3 setup.py develop
python3 scripts/MarsBest.py --config-file configs/bigtosmalllong.yaml --transforms random_flip random_erase --root /home2/lgfm95/reid/uavdata/ --save_path "bigibn.pth" --ncc True --epochs 500 --model_path none > bigibn.1.out 2> bigibn.1.err
python3 scripts/MarsBest.py --config-file configs/bigtosmalllong.yaml --transforms random_flip random_erase --root /home2/lgfm95/reid/uavdata/ --save_path "bigibn.pth" --ncc True --epochs 500 --model_path none > bigibn.2.out 2> bigibn.2.err
python3 scripts/MarsBest.py --config-file configs/bigtosmalllong.yaml --transforms random_flip random_erase --root /home2/lgfm95/reid/uavdata/ --save_path "bigibn.pth" --ncc True --epochs 500 --model_path none > bigibn.3.out 2> bigibn.3.err
