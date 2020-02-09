# ObjectDetectionSystem
Object Detection System.

Detect objects from video or image.

Can use different models such as   ssd_inception_v2_coco_2018_01_28, faster_rcnn_inception_v2_coco_2018_01_28 and similar. 

-----------------------------------------------------------------------------------------------------------------------------
All used models are coco models which means they can detect classes such as :
person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,stop sign,parking meter,bench,bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier and toothbrush. 
The difference between models is accuracy and speed. 

Custom built model for learning purposes can also be used in this system. It includes 3 objects:
 (blue_duck originally "plava_patkica", CASIO_fx_991ES and CASIO_fx_991ES_PLUS). 
 
-----------------------------------------------------------------------------------------------------------------------------

Technologies: Tensorflow, Python, opencv, matplotlib.

-----------------------------------------------------------------------------------------------------------------------------

# Example usage -demo
-----------------------------------------------------------------------------------------------------------------------------

Main screen - choosing detection model, choosing between video/image detection mode.
![1](https://user-images.githubusercontent.com/26230313/74101454-3eae0600-4b3a-11ea-9c55-783c5f4247ec.PNG)

Video detection examples

![2](https://user-images.githubusercontent.com/26230313/74101468-7f0d8400-4b3a-11ea-8f59-ba9d2bb0d309.PNG)
![3](https://user-images.githubusercontent.com/26230313/74101472-86cd2880-4b3a-11ea-85f5-e09ae10b90a7.PNG)

Image detection examples
![4](https://user-images.githubusercontent.com/26230313/74101477-964c7180-4b3a-11ea-9615-b826b3d45bd4.PNG)
![5](https://user-images.githubusercontent.com/26230313/74101478-9a788f00-4b3a-11ea-82be-a9da57e6f541.PNG)

Mask output model example
![6](https://user-images.githubusercontent.com/26230313/74101488-a9f7d800-4b3a-11ea-80bd-e16c7d3a730a.PNG)

Precision comparison on same image between different models 
![7](https://user-images.githubusercontent.com/26230313/74101509-dc093a00-4b3a-11ea-8d77-9096418a7bfe.PNG)
-----------------------------------------------------------------------------------------------------------------------------

# Some configuration notes

Since this is not buildable release version, I'll put here some configuration Tensorflow notes that might be helpful in future, that I used before creating project:

cloning Tensorflow
git clone https://github.com/tensorflow/models

from models/research/
python3 setup.py build
sudo python3 setup.py install

installing dependencies

pip install --user Cython
pip install --user contextlib2
pip install --user pillow
pip install --user lxml
pip install --user jupyter
pip install --user matplotlib

check that installation paths are in PATH environment variable

cloning models directory 
git clone https://github.com/tensorflow/models.git

COCO API installation 

git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools <path_to_tensorflow>/models/research/

.proto files to .py conversion using protoc
sudo ../../Downloads/bin/protoc object_detection/protos/*.proto --python_out=.
sudo python3 use_protobuf.py object_detection/protos /home/yellow/Downloads/bin/protoc

downloading Protobuf
https://github.com/protocolbuffers/protobuf/releases

adding slim and research directories to environment variables 

export PYTHONPATH=$PYTHONPATH:/models/research
export PYTHONPATH=$PYTHONPATH:/models/research/object_detection
export PYTHONPATH=$PYTHONPATH:/models/research/slim

opencv installation 
pip install opencv-python==3.4.4.19


Used version 1.14
sudo python3 -m pip install tensorflow==1.14
sudo python3 -m pip install tensorflow-gpu==1.14

-----------------------------------------------------------------------------------------------------------------------------
Following error 
ModuleNotFoundError: No module named 'nets'
means path is not configured well 
Fix with adding slim to path -> from directory  /models/research$ export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

-----------------------------------------------------------------------------------------------------------------------------

Main file for runing this system is 
ObjectDetectionSystem.py


