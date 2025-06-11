docker run -it --rm --gpus all --name uniscene \
  --privileged \
  --shm-size=8g \
  -v /storage_local/kwang/repos/UniScene:/UniScene \
  -v /storage_local/kwang/UniScene_data/occ:/UniScene/occupancy_gen/data/gts \
  -v /storage_local/kwang/UniScene_data/nuscenes_infos_train_temporal_v3_scene.pkl:/UniScene/occupancy_gen/data/nuscenes_infos_train_temporal_v3_scene.pkl \
  -v /storage_local/kwang/UniScene_data/nuscenes_infos_val_temporal_v3_scene.pkl:/UniScene/occupancy_gen/data/nuscenes_infos_val_temporal_v3_scene.pkl \
  -v /storage_local/kwang/UniScene_data/step2:/UniScene/occupancy_gen/data/step2 \
  -v /storage_local/kwang/UniScene_data/gen_occ:/UniScene/occupancy_gen/gen_occ \
  -v /storage_local/kwang/UniScene_data/ckpt:/UniScene/occupancy_gen/ckpt \
  -v /mrtstorage/datasets/public/nuscenes.sqfs:/data/nuscenes.sqfs \
  -w /UniScene \
  uniscene:local \
  bash -c "./prepare_data.sh && exec bash"