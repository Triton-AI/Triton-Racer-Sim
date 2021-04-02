#!/bin/bash
docker run --name tritonracer --rm -it \
    --mount type=volume,source=tritonracer,target=/projects/c1 \
    --network=host \
    --privileged \
    --runtime=nvidia \
    --gpus=all \
    haoru233/tritonai-tritonracer:tf2.3-jp4.5