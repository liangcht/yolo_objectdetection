FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.9.0-cudnn.7
# Copy the files to the necessary folder
RUN mkdir /home/job
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod a+rwx /home/job/toolkit-execute
COPY envs.sh /home/job/envs.sh

# Labels for the docker
LABEL description="This docker has MSR CCS Caffe and python2" \
      repository="philly/jobs/custom/msrccs-caffe" \
      tag="nv017" \
      creator="ehazar" tooltype="caffe" \
      createtime="5/1/2018"

# sane bash defaults
COPY bashrc.sh /etc/
RUN chmod 655 /etc/bashrc.sh
RUN echo "source /etc/bashrc.sh" >> /etc/bash.bashrc

# Prepare Caffe environment
ENV CAFFE_ROOT=/opt/caffe
RUN mkdir -p $CAFFE_ROOT/lib

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
RUN mkdir -p $PYCAFFE_ROOT

ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/bin:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/lib" >> /etc/ld.so.conf.d/caffe.conf

RUN apt-get -y update
RUN apt-get install -y \
      python-pip

# upgrade pip
RUN pip install --upgrade pip

## install development requirements
RUN apt-get install -y libprotobuf-dev protobuf-compiler
RUN apt-get install -y libatlas-base-dev libboost-all-dev
RUN apt-get install -y libgflags-dev libgoogle-glog-dev
RUN apt-get install -y libhdf5-serial-dev
RUN apt-get install -y libleveldb-dev liblmdb-dev
RUN apt-get install -y libopencv-dev
RUN apt-get install -y libsnappy-dev
RUN apt-get install -y python-dev
RUN apt-get install -y python-qt4
RUN apt-get install -y libyaml-dev
RUN apt-get install -y libturbojpeg
RUN apt-get install -y vim
RUN apt-get install -y git
RUN apt-get install -y tmux

RUN apt-get autoremove

# NCCL2
COPY nccl-repo-ubuntu1604-2.1.15-ga-cuda9.0_1-1_amd64.deb /home/job
RUN dpkg -i /home/job/nccl-repo-ubuntu1604-2.1.15-ga-cuda9.0_1-1_amd64.deb

#tmux conf
COPY tmux.conf /etc/

# pip packages
COPY requirements.txt /home/job
RUN pip install -r /home/job/requirements.txt

ENV NLTK_DATA /home/job/nltk_data
RUN mkdir -p $NLTK_DATA
RUN python -m nltk.downloader wordnet brown averaged_perceptron_tagger

# To build in the docker directly (make sure /src/Caffe has the source):
#RUN mkdir -p /src/Caffe/build
#RUN cd /src/Caffe/build && \
#    cmake -DUSE_MPI=1 -DUSE_OPENMP=1 -DUSE_CUDNN=1 -DUSE_NCCL=1 -DCUDA_ARCH_NAME=All -DCMAKE_INSTALL_PREFIX=$CAFFE_ROOT -DBLAS=atlas .. && \
#    make -j$(nproc) && make install
# To prebuild .tar.gz file:
# make DESTDIR=/caffe_install install
# tar -zcvf /src/caffe.tar.gz /caffe_install
COPY caffe.tar.gz /home/job
RUN tar --strip-components=1 -C / -xvf /home/job/caffe.tar.gz

# Make it possible to modify this if needed
RUN chmod -R 777 /opt/

# remove intermediates
RUN rm -rf /home/job/*.tar.gz
RUN rm -rf /home/job/*.deb

# To Build:
# docker build -t msrccs-caffe .
# To clean up older images:
#  0. If docker > 1.13: docker system prune
#  1. Stop all exited containers
#   docker rm $(docker ps -a -q -f "status=exited")
#  2. Remove all dangling images (failed builds, ...)
#   docker rmi $(docker images -aq --filter dangling=true)
# To develop on Windows (note the mount: /src):
#  winpty docker run -v d:\\development\\:/src --rm -ti msrccs-caffe
