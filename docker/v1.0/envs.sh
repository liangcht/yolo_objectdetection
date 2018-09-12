#!/bin/bash

export CAFFE_ROOT=${CAFFE_ROOT:-/opt/caffe}
export PYCAFFE_ROOT=${PYCAFFE_ROOT:-$CAFFE_ROOT/python}
export PYTHONPATH=${PYTHONPATH:-$PYCAFFE_ROOT:$PYTHONPATH}
export NLTK_DATA=${NLTK_DATA:-/home/job/nltk_data}
