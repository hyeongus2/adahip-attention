#!/usr/bin/env bash

docker run -it --gpus all \
  --name acr \
  --shm-size=2g \
  -v "$PWD:/work" -w /work \
  deepauto/hip-attention:latest \
  bash
