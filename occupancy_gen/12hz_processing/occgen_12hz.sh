# save 12hz 200*200 bevlayout
python ./12hz_processing/save_bevlayout_12hz.py

#infer 12hz occupancy with OccDiT training on Occ3D
CUDA_VISIBLE_DEVICES=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=12345 python ./12hz_processing/eval_12hz_with_occ3d.py
