#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
##SBATCH -p gpu-private
#SBATCH -p res-gpu-small
#SBATCH --qos=short
#SBATCH --job-name=tempvit
#SBATCH --gres=gpu:1
#SBATCH -o tempvit.out
#SBATCH --mem=24g
#SBATCH -t 2-0:0:0

source ../venv/bin/activate

#pip3 install -r deep-person-reid/requirements.txt
#python3 setup.py develop
python3 scripts/MarsBest.py --config-file configs/temporallynearlong.yaml --transforms random_flip random_erase --root /home2/lgfm95/reid/uavdata/ --save_path "tempvit.pth" --ncc True --epochs 500 --model_name backbone --model_path none > tempvit.1.out 2> tempvit.1.err
python3 scripts/MarsBest.py --config-file configs/temporallynearlong.yaml --transforms random_flip random_erase --root /home2/lgfm95/reid/uavdata/ --save_path "tempvit.pth" --ncc True --epochs 500 --model_name backbone --model_path none > tempvit.2.out 2> tempvit.2.err
python3 scripts/MarsBest.py --config-file configs/temporallynearlong.yaml --transforms random_flip random_erase --root /home2/lgfm95/reid/uavdata/ --save_path "tempvit.pth" --ncc True --epochs 500 --model_name backbone --model_path none > tempvit.3.out 2> tempvit.3.err
