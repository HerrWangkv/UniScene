docker run -it --rm --gpus all --name uniscene \
  --privileged \
  --shm-size=8g \
  -v /storage_local/kwang/repos/UniScene:/UniScene \
  -v /storage_local/kwang/nuscenes/occ:/UniScene/occupancy_gen/data/gts \
  -v /storage_local/kwang/nuscenes/nuscenes_infos_train_temporal_v3_scene.pkl:/UniScene/occupancy_gen/data/nuscenes_infos_train_temporal_v3_scene.pkl \
  -v /storage_local/kwang/nuscenes/nuscenes_infos_val_temporal_v3_scene.pkl:/UniScene/occupancy_gen/data/nuscenes_infos_val_temporal_v3_scene.pkl \
  -v /mrtstorage/datasets/public/nuscenes.sqfs:/data/nuscenes.sqfs \
  -w /UniScene \
  uniscene:local \
  bash -c "./prepare_data.sh && exec bash"