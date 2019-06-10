# Multi-Object detection using DeepLearning

## Description
This is project C, training an object detection classifier using tensorflow API on Windows 10.
## Requirements
============
Install package 'tensorflow' as follow:
```
$ pip install tensorflow
```
Install package 'tensorflow-gpu' as follow:
```
$ pip install tensorflow-gpu
```
Install package 'pycocotools' as follow:
```
$ pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
```
## Code organization
demo.ipynb -- Run a demo of our code: Using 2 different trained models to perform object detection on 2 photos.<br>
create_pascal_tfrecord.py -- Transofrm original images to data that can be manipulated in tensorflow.<br>
MODEL_NAME.config -- Experimental setting for each different model.<br>
train.py -- Run the training process.<br>
eval.py -- Run the evaluation process.<br>
## Guidance
=================
# 1.Download VOC2012
# 2.Transform data
-----------------
    REMINDER: All the commands below should be run in cmd.exe in the directory of WORKSPACE.
    Here is an example:
    >WORKSPACE
    >>object_detection
    >>>create_pascal_tfrecord.pyy
    >>>train.py
    >>>eval.py
    >>>voc
Running 'create_pascal_tfrecord.py' as follow:
$ python object_detection/create_pascal_tf_record.py 
--data_dir=object_detection/voc/VOCdevkit/ 
--year=VOC2012 
--set=train 
--output_path=object_detection/voc/pascal_train.record

$ python object_detection/create_pascal_tf_record.py 
--data_dir=object_detection/voc/VOCdevkit/ 
--year=VOC2012 
--set=val 
--output_path=object_detection/voc/pascal_val.record
