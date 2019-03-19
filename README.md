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

To build the extensions and *install* the `mmod` and `mtorch` packages:

```bash
python setup.py develop
```

With `develop` you do not need to call the above again when changing `.py` 
files. But if you change a cpp extension you will have to call this again to build. 
If you do not want to develop the package and just want to install:
 
```bash
python setup.py install
```

For philly, you can `install` only the cpp extensions inside your docker, and then call the scripts.
All scripts (that have `main`) append the parent directory to the `sys.path` and can be run without installation. 

Usage Wiki
==========
Refer to the project [wiki](https://github.com/leizhangcn/objectdetection/wiki) for more usage examples.
