#!/bin/bash

set -x
cwd=$(dirname "${BASH_SOURCE[0]}")

export DATA_DIR
export MODEL_DIR
export LOG_DIR
export PREV_MODEL_PATH
export STDOUT_DIR

OMPI_COMM_WORLD_LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK:-0}
OMPI_COMM_WORLD_RANK=${OMPI_COMM_WORLD_RANK:-0}
printenv > ~/mpi_env_$OMPI_COMM_WORLD_RANK
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

if [[ $OMPI_COMM_WORLD_RANK == 0 ]]; then
    # for philly-on-ap hack
    if [[ $PHILLY_CLUSTER == philly-prod-cy4 ]]; then
        cp ~/.ssh/id_rsa* ~/ && true
        sudo chmod 777 ~/id_rsa* && true
    fi
fi

newCaffe="$cwd/caffe.tar.gz"

if [[ -f "$newCaffe" ]]; then
    sudo tar --strip-components=1 -C / -xvf $newCaffe
fi

if [[ $OMPI_COMM_WORLD_RANK == 0 && ! -f ~/.bash_history ]]; then
    if [[ -f "$cwd"/bash_history ]]; then
        cp $cwd/bash_history ~/.bash_history
    fi
fi

# work around PhillyOnMP issue by not installing mpi4y if world size is 1
#if [[ ${OMPI_COMM_WORLD_SIZE:-1} > 1 ]]; then
#sudo pip install mpi4py
#fi

#sudo apt-get install -y git tmux
#sudo pip install pyyaml progressbar easydict ete2
#sudo pip install nltk
#python -m nltk.downloader all

touch /tmp/ready
