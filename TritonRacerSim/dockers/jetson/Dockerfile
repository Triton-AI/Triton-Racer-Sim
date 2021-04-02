FROM nvcr.io/nvidia/l4t-tensorflow:r32.5.0-tf2.3-py3

# Install softwares
RUN apt-get update -y && apt-get upgrade -y
RUN apt-get install -y python3-dev python3-pip
RUN apt-get install -y git nano rsync zip
RUN apt-get install -y libsdl-image1.2-dev libsdl-mixer1.2-dev \
    libsdl-ttf2.0-dev libsdl1.2-dev libsmpeg-dev python-numpy \
    subversion libportmidi-dev ffmpeg libswscale-dev \
    libavformat-dev libavcodec-dev libfreetype6-dev

# Install Python packages
# SHELL [ "/bin/bash", "-c" ]
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -U docopt pyserial \
    pillow keras pygame==1.9.6 opencv-python simple-pid \
    scikit-learn

RUN mkdir -p /projects
WORKDIR /projects
RUN git clone https://github.com/LiamBindle/PyVESC && \
    cd PyVESC; python3 -m pip install -e .

RUN git clone https://github.com/Triton-AI/Triton-Racer-Sim && \
    cd Triton-Racer-Sim && \
    git checkout dev && \
    cp -r TritonRacerSim/car_templates /projects/ && \
    echo "export PYTHONPATH=${PYTHONPATH}:/projects/Triton-Racer-Sim/" >> ~/.bashrc

RUN mv car_templates c1
WORKDIR /projects/c1
