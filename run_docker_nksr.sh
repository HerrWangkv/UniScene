docker run -it --rm --gpus all --name uniscene-nksr \
  --privileged \
  --shm-size=8g \
  -v /storage_local/kwang/repos/UniScene:/UniScene \
  -v /storage_local/kwang/UniScene_data/data:/UniScene/data \
  -w /UniScene \
  uniscene-nksr:local