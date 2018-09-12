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

newCaffe="$cwd/caffe.tar.gz"

if [[ -f "$newCaffe" ]]; then
    sudo tar --strip-components=1 -C / -xvf $newCaffe
fi

if [[ -f "$cwd"/bash_history ]]; then
    cp $cwd/bash_history ~/.bash_history
fi

#sudo apt-get install git tmux
#sudo pip install pyyaml progressbar easydict ete2
#sudo pip install nltk
#python -m nltk.downloader all

touch /tmp/ready
