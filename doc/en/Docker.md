# Docker

## Prerequisites
* Docker must be installed and running on your system.
* Create a folder to store big models & intermediate files (ex. /mnt/models)

## Images
There are Docker images available for our project：

**Uploading**

## Building docker image locally
 - Download Dockerfile in [there](../../Dockerfile)

 - finish, execute
   ```bash
   docker build  -t approachingai/ktransformers:v0.1.1 .
   ```

## Usage

Assuming you have the [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) that you can use the GPU in a Docker container.
```
docker run --gpus all -v /path/to/models:/models -p 10002:10002 approachingai/ktransformers:v0.1.1 --port 10002 --gguf_path /models/path/to/gguf_path --model_path /models/path/to/model_path --web True
```

More operators you can see in the [readme](../../README.md)