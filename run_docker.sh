docker run -it --rm --gpus all --name uniscene \
  --privileged \
  --shm-size=8g \
  -v /storage_local/kwang/repos/UniScene:/UniScene \
  -v /storage_local/kwang/UniScene_data/data:/UniScene/data \
  -v /storage_local/kwang/UniScene_data/gen_occ:/UniScene/gen_occ \
  -v /storage_local/kwang/UniScene_data/ckpt:/UniScene/ckpt \
  -w /UniScene \
  uniscene:local