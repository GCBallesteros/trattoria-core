# Building and Publishing to Pypi

Windows and Mac x64 wheels are built and published using Github actions. On the other
hand Linux and Mac Silicon wheels are build docker using a Docker container or just a
Mac laptop (with M-series chips).

# Linux
1. Build the docker container with `docker build -t wheels . --file DOCKERFILE` from the
   linux folder. This will create a ManyLinux container.
2. The start the container with `docker run -it --entrypoint=/bin/bash wheels`.
3. Go to the `/root` folder and run the `build-wheels.sh` script followed by the
   `publish.sh` script. The latter will ask for your Pypi user and password.
4. Once you exit the container everything will be destroyed and cleaned up.

# Mac Silicon
Some steps as above but the have to be run from a Mac Silicon machine. Remember to start
a virtual environment with Maturin installed in it. Right now the script only works for
Python3.10.
