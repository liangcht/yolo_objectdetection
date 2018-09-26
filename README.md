Microsoft Massive Object Detection
------------


Introduction
============

Utilities to create create, explore, and train objectdetection taxonomies.

License
============
Microsoft Object Detection is copyright Microsoft.

Installation
============

1. Install Anaconda2
    ```
    wget --quiet https://repo.anaconda.com/archive/Anaconda2-5.2.0-Linux-x86_64.sh -O ~/anaconda.sh
    /bin/bash ~/anaconda.sh -b -p /opt/conda
    rm ~/anaconda.sh
    ```
    
    i. If you have Anaconda (even Anaconda3), you can set up Python2 virtual environment with:
        
    ```
    conda create --name py27 python=2.7 anaconda
    ```

2. Use Anaconda Python
    ```
    export PATH="/opt/conda/bin:$PATH"
    ```
    optionally, make it default for all the sessions
    ```
    echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh
    ```
    
    i. If you are using a virtual environment (e.g. `py27`)

    ```
    source activate py27
    ```
  
3. Install requirements
    ```
    conda install pytorch torchvision -c pytorch
    pip install -r requirements.txt
    ```

4. Development
    
    To build the extensions and *install* the `mmod` and `mtorch` packages:

    ```
    python setup.py develop
    ```
    With `develop` you do not need to call the above again when changing `.py` 
    files. But if you change a cpp extension you will have to call this again to build. 
    If you do not want to develop the package and just want to install: 
    ```
    python setup.py install
    ```
    For philly, you can `install` only the cpp extensions inside your docker, and then call the scripts.
    All scripts (that have `main`) append the parent directory to the `sys.path` and can be run without installation. 

Caffe setup
===========
If you want to also install caffe for Anaconda, you should re-build opencv suitable for it. 
   Refer to [pt04/Dockerfile](docker/pt04/Dockerfile) to see how to build opencv library.
   Note that caffe is currently needed as PyTorch's data reader.
   
   i. build opencv as instructed
    
   ii. Reference Anaconda Python when building caffe.
   ```
   cmake -DUSE_CUDNN=1 -DUSE_NCCL=1 -DUSE_OPENMP=1 -DCMAKE_INSTALL_PREFIX=/opt/caffe -DPYTHON_LIBRARY=/opt/conda2/lib/libpython2.7.so -DPYTHON_EXECUTABLE:FILEPATH=/opt/conda2/bin/python2.7 ..
   make -j$(nproc)
   sudo make install 
   ```
   Note: Use `-DUSE_CUDNN=0 -DUSE_NCCL=0` to build caffe without cuda support. 
   If you only want caffe for PyTorch data layer this will be sufficient. 
 
   iii. Use `export PYTHONPATH=/opt/caffe/python/` to be able to import caffe.
   
   iv. Use `export PATH=/opt/caffe/bin:/opt/caffe:$PATH` to be able to use `runcaffe` (to see caffe executable)

Usage Wiki
==========
Refer to the project [wiki](https://github.com/leizhangcn/objectdetection/wiki) for more usage examples.
