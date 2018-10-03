FROM phillyregistry.azurecr.io/philly/jobs/base/philly-openmpi:1.10.3-ubuntu.16.04-cuda.8.0-cudnn.6

# Copy the files to the necessary folder
RUN mkdir /home/job
COPY toolkit-execute /home/job/toolkit-execute
RUN chmod a+rwx /home/job/toolkit-execute
COPY envs.sh /home/job/envs.sh


# Labels for the docker
LABEL description="This docker contains PyTorch 0.4 using python 2.7 from anaconda and MSRCCS/caffe " \
      repository="philly/jobs/custom/msrccs-caffe" \
      tag="pt04" \
      creator="ehazar" tooltype="pytorch" \
      tooltypeversion="0.4.0" \
      createtime="6/5/2018"

RUN apt-get -y update
RUN apt-get install -y vim
RUN apt-get install -y tmux

# tmux conf
COPY tmux.conf /etc/

# sane bash defaults
COPY bashrc.sh /etc/
RUN chmod 655 /etc/bashrc.sh
RUN echo "source /etc/bashrc.sh" >> /etc/bash.bashrc

# Make it possible to modify this if needed (/opt will be shared fo users)
RUN chmod -R 777 /opt
RUN chgrp -Rv users /opt
RUN find /opt -type d -exec chmod g+s '{}' \;

# Install Anaconda for python 2.7
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.anaconda.com/archive/Anaconda2-5.2.0-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh
# regular user to be able to install new package here
RUN chmod 777 /opt/conda/lib/python2.7/site-packages

# pip packages
COPY requirements.txt /home/job
RUN /opt/conda/bin/pip install -r /home/job/requirements.txt


# Prepare environment (Caffe, Cuda)
ENV CAFFE_ROOT=/opt/caffe
RUN mkdir -p $CAFFE_ROOT/lib
ENV PYCAFFE_ROOT $CAFFE_ROOT/python
RUN mkdir -p $PYCAFFE_ROOT
ENV PATH /opt/conda/bin:$CAFFE_ROOT/bin:$PYCAFFE_ROOT:$PATH
ENV NLTK_DATA /home/job/nltk_data
RUN export CUDA_HOME="/usr/local/cuda"
ENV CUDA_ROOT /usr/local/cuda/bin
ENV CUDA_HOME /usr/local/cuda

# Install pytorch GPU version.
RUN conda install pytorch=0.4.0 torchvision -c pytorch

# Python Caffe
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
RUN echo "$CAFFE_ROOT/lib" >> /etc/ld.so.conf.d/caffe.conf

# Created with: tar cfz mtorch.tar.gz mtorch
# Install our extensions (they can be updated later through pre-run)
COPY mtorch /home/job/mtorch
RUN chmod -R 777 /home/job

# Fixed in upstream
RUN sed -i 's/CUDA_HOME = _find_cuda_home()/CUDA_HOME = _find_cuda_home()#/g' /opt/conda/lib/python2.7/site-packages/torch/utils/cpp_extension.py

# install just for caffe and its opencv
RUN apt-get install -y libprotobuf-dev protobuf-compiler
RUN apt-get install -y libatlas-base-dev libboost-all-dev
RUN apt-get install -y libgflags-dev libgoogle-glog-dev
RUN apt-get install -y libhdf5-serial-dev
RUN apt-get install -y libleveldb-dev liblmdb-dev

# build opencv
RUN apt-get install -y unzip
RUN wget https://github.com/Itseez/opencv/archive/3.3.0.zip -O /home/job/ocv.zip && \
  unzip /home/job/ocv.zip && cd opencv-3.3.0 && mkdir build && cd build && \
  cmake -DWITH_CUDA=OFF -DBUILD_TIFF=ON -DBUILD_opencv_java=OFF -DENABLE_AVX=ON -DWITH_OPENGL=OFF -DWITH_OPENCL=OFF -DWITH_IPP=ON \
  -DWITH_TBB=ON -DWITH_EIGEN=ON -DWITH_V4L=ON -DWITH_VTK=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DCMAKE_BUILD_TYPE=RELEASE \
  -DBUILD_opencv_python3=OFF -DBUILD_opencv_python2=ON -DPYTHON_EXECUTABLE=/opt/conda/bin/python2.7 -DPYTHON_LIBRARY=/opt/conda/lib/libpython2.7.so \
  -DPYTHON_INCLUDE_DIR=/opt/conda/include/python2.7 -DPYTHON_PACKAGES_PATH=/opt/conda/lib/python2.7/site-packages \
  -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/opt/conda -DINSTALL_PYTHON_EXAMPLES=OFF -DINSTALL_C_EXAMPLES=OFF \
  -DBUILD_EXAMPLES=OFF .. && \
  make -j$(nproc) && make install && \
  cd ../../ && rm -rf opencv-3.3.0

# To build in the docker directly (make sure /src/Caffe has the source):
#RUN mkdir -p /src/Caffe/build
#RUN cd /src/Caffe/build && \
#    cmake -DUSE_OPENMP=1 -DPYTHON_LIBRARY=/opt/conda/lib/libpython2.7.so -DPYTHON_EXECUTABLE:FILEPATH=/opt/conda/bin/python2.7 -DUSE_CUDNN=0 -DUSE_NCCL=0 -DBLAS=atlas -DCMAKE_INSTALL_PREFIX=$CAFFE_ROOT .. \
#    make -j$(nproc) && make install
# To prebuild .tar.gz file:
# make DESTDIR=/caffe_install install
# tar -zcvf /src/caffe.tar.gz /caffe_install
COPY caffe.tar.gz /home/job
RUN tar --strip-components=1 -C / -xvf /home/job/caffe.tar.gz

ENV MPLCONFIGDIR /home/job/.config/matplotlib
ENV PYTHON_EGG_CACHE /tmp

RUN  cd /home/job/mtorch/rt && /opt/conda/bin/pip install .
RUN  cd /home/job/mtorch/smt && /opt/conda/bin/pip install .
RUN  cd /home/job/mtorch/smtl && /opt/conda/bin/pip install .

# remove intermediates
RUN rm -rf /home/job/*.tar.gz
RUN rm -rf /home/job/*.deb
RUN rm -rf /home/job/*.zip

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
