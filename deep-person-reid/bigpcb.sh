#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
##SBATCH -p gpu-private
#SBATCH -p res-gpu-small
#SBATCH --qos=short
#SBATCH --job-name=bigpcb
#SBATCH --gres=gpu:1
#SBATCH -o bigpcb.out
#SBATCH --mem=24g
#SBATCH -t 2-0:0:0

source ../venv/bin/activate

#pip3 install -r deep-person-reid/requirements.txt
#python3 setup.py develop
python3 scripts/MarsBest.py --config-file configs/bigtosmallpcb.yaml --transforms random_flip random_erase --root /home2/lgfm95/reid/uavdata/ --save_path "bigpcb.pth" --ncc True --epochs 500 --model_name backbone --model_path none > bigpcb.1.out 2> bigpcb.1.err
python3 scripts/MarsBest.py --config-file configs/bigtosmallpcb.yaml --transforms random_flip random_erase --root /home2/lgfm95/reid/uavdata/ --save_path "bigpcb.pth" --ncc True --epochs 500 --model_name backbone --model_path none > bigpcb.2.out 2> bigpcb.2.err
python3 scripts/MarsBest.py --config-file configs/bigtosmallpcb.yaml --transforms random_flip random_erase --root /home2/lgfm95/reid/uavdata/ --save_path "bigpcb.pth" --ncc True --epochs 500 --model_name backbone --model_path none > bigpcb.3.out 2> bigpcb.3.err
