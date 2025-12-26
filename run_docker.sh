#!/usr/bin/env bash
set -euo pipefail

NAME="acr"
IMAGE="deepauto/hip-attention:latest"

# If container doesn't exist, create it once
if ! docker ps -a --format '{{.Names}}' | grep -qx "${NAME}"; then
  echo "[+] Creating container: ${NAME}"
  docker run -it \
    --name "${NAME}" \
    --gpus all \
    -v "$PWD:/work" -w /work \
    "${IMAGE}" \
    bash
  exit 0
fi

# If it exists, start (if needed) and exec into it
if ! docker ps --format '{{.Names}}' | grep -qx "${NAME}"; then
  echo "[+] Starting container: ${NAME}"
  docker start "${NAME}" >/dev/null
fi

echo "[+] Entering container: ${NAME}"
docker exec -it "${NAME}" bash
