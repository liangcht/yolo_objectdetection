#!/bin/bash

cwd=$(dirname "${BASH_SOURCE[0]}")

OMPI_COMM_WORLD_LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK:-0}
if [[ $OMPI_COMM_WORLD_LOCAL_RANK != 0 ]]; then
    return
fi

if [[ -f ~/.bash_history ]]; then
    cp ~/.bash_history $cwd/bash_history || true
fi
