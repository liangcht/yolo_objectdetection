#!/bin/bash

set -x
cwd=$(dirname "${BASH_SOURCE[0]}")

OMPI_COMM_WORLD_LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK:-0}
#printenv > /tmp/mpi_env_$OMPI_COMM_WORLD_LOCAL_RANK
if [[ $OMPI_COMM_WORLD_LOCAL_RANK != 0 ]]; then
    counter=0
    while [[ ! -f /tmp/ready ]]; do
      sleep 2
    done
    if [[ "$counter" -gt 120 ]]; then
        echo "Timeout waiting for master"
        exit 1
    fi
    return
fi

cp ~/.ssh/id_rsa* ~/ && true
sudo chmod 777 ~/id_rsa* && true

newCaffe="$cwd/caffe.tar.gz"

if [[ -f "$newCaffe" ]]; then
    sudo tar --strip-components=1 -C / -xvf $newCaffe
fi

if [[ -f "$cwd"/bash_history ]]; then
    cp $cwd/bash_history ~/.bash_history
fi

# work around PhillyOnMP issue by not installing mpi4y if world size is 1
if [[ ${OMPI_COMM_WORLD_SIZE:-1} > 1 ]]; then
    sudo pip install $cwd/mpi4py-3.0.0-cp27-cp27mu-linux_x86_64.whl
fi

sudo chmod 0666 /dev/nvidia* && true

touch /tmp/ready

#sudo pip install mpi4py
#sudo apt-get install -y git tmux
#sudo pip install pyyaml progressbar easydict ete2
#sudo pip install nltk
#python -m nltk.downloader all
