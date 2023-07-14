docker build -t wheels . --file DOCKERFILE
sudo docker run -it --entrypoint=/bin/bash wheels
