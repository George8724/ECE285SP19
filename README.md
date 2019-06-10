# Multi-Object detection using DeepLearning

## Description
This is project C, training a multi-object detection classifier using tensorflow API on Windows 10.

## Requirements
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
Install CUDA v10.0 and CUDNN following the official instructions.

## Code organization
demo.ipynb -- Run a demo of our code: Using 2 different trained models to perform object detection on 2 photos.<br><br>
create_pascal_tfrecord.py -- Transofrm original images to data that can be manipulated in tensorflow.<br><br>
MODEL_NAME.config -- Experimental setting for each different model.<br><br>
train.py -- Run the training process.<br><br>
eval.py -- Run the evaluation process.

## Guidance
    REMINDER: All the commands below should be run on cmd in the directory of WORKSPACE.

Here is an example:
>WORKSPACE
>>object_detection
>>>create_pascal_tfrecord.py<br>
>>>train.py<br>
>>>eval.py<br>
>>>voc
>>>>config<br>
>>>>train_dir<br>
>>>>eval<br>
>>>>pretrained<br><br>

In this project, we performed 3 different models. In the following instructions we will use ssd as example.

### 1.Run 'demo.ipynb'
First, download our trained models from the link below:
ssd:https://drive.google.com/file/d/1sChpTiWNYI9M0cT5WXNoZdLL30rkzg2r/view?usp=sharing <br>
faster-cnn:https://drive.google.com/file/d/12O7B-ZDwYd5OiW7BTnfD6hDp1jQIEqZg/view?usp=sharing <br>
Then run 'demo.ipynb'.
### 2.Download VOC2012
Download the VOC2012 image set then put all the files into WORKSPACE/object_detection/voc/VOCdevkit.

### 3.Transform data and generate train/test sets

Run 'create_pascal_tfrecord.py' as follow:
```
$ python object_detection/create_pascal_tf_record.py 
--data_dir=object_detection/voc/VOCdevkit/ 
--year=VOC2012 
--set=train 
--output_path=object_detection/voc/pascal_train.record
```
```
$ python object_detection/create_pascal_tf_record.py 
--data_dir=object_detection/voc/VOCdevkit/ 
--year=VOC2012 
--set=val 
--output_path=object_detection/voc/pascal_val.record
```

### 4.Download pretrained models
Go to the links below:
http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017.tar.gz
http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_11_06_2017.tar.gz
http://download.tensorflow.org/models/object_detection/rfcn_resnet101_coco_11_06_2017.tar.gz

Unzip them and then put all the files into WORKSPACE/object_detection/voc/pretrained/MODEL_NAME.<br>
For example,
>pretrained
>>ssd
>>>frozen_inference_graph.pb<br>
>>>graph.pbtxt<br>
>>>model.ckpt.data-00000-of-00001<br>
>>>model.ckpt.index<br>
>>>model.ckpt.meta

### 5.Train the models
Run 'train.py' as follow:
```
$ python object_detection/train.py 
--train_dir=object_detection/voc/train_dir/ssd/ 
--pipeline_config_path=object_detection/voc/config/ssd.config 
```

### 6.Evaluate the results
Run 'eval.py' as follow:
```
$ python object_detection/eval.py 
--logtostderr 
--checkpoint_dir=object_detection/voc/train_dir/ssd/ --eval_dir=object_detection/voc/eval/ssd/
--pipeline_config_path=object_detection/voc/config/ssd.config
```

