#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
##SBATCH -p gpu-private
#SBATCH -p res-gpu-small
#SBATCH --qos=short
#SBATCH --job-name=bigmlfn
#SBATCH --gres=gpu:1
#SBATCH -o bigmlfn.out
#SBATCH --mem=24g
#SBATCH -t 2-0:0:0

source ../venv/bin/activate

#pip3 install -r deep-person-reid/requirements.txt
#python3 setup.py develop
python3 scripts/MarsBest.py --config-file configs/bigtosmallmlfn.yaml --transforms random_flip random_erase --root /home2/lgfm95/reid/uavdata/ --save_path "bigmlfn.pth" --ncc True --epochs 500 --model_name backbone --model_path none > bigmlfn.1.out 2> bigmlfn.1.err
python3 scripts/MarsBest.py --config-file configs/bigtosmallmlfn.yaml --transforms random_flip random_erase --root /home2/lgfm95/reid/uavdata/ --save_path "bigmlfn.pth" --ncc True --epochs 500 --model_name backbone --model_path none > bigmlfn.2.out 2> bigmlfn.2.err
python3 scripts/MarsBest.py --config-file configs/bigtosmallmlfn.yaml --transforms random_flip random_erase --root /home2/lgfm95/reid/uavdata/ --save_path "bigmlfn.pth" --ncc True --epochs 500 --model_name backbone --model_path none > bigmlfn.3.out 2> bigmlfn.3.err
