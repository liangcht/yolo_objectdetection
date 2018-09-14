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

    ```
    python setup.py develop
    ```
    With `develop` you do not need to call the above again when changing `.py` 
    files. But if you change an extension you will have to call this again to build.

Caffe
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
 
   iii. Use `export PYTHONPATH=/opt/caffe/python/` to be able to import caffe.
