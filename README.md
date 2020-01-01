# ObjectDetectionSystem
Object Detection System.
Detect objects from video or image.
Can use different models such as   ssd_inception_v2_coco_2018_01_28, faster_rcnn_inception_v2_coco_2018_01_28 and similar. 
Also contains custom built model for learning purposes which includes 3 objects (blue_duck originally "plava_patkica", CASIO_fx_991ES and CASIO_fx_991ES_PLUS). 

Technologies: Tensorflow, Python, opencv, matplotlib.

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

Following error 
ModuleNotFoundError: No module named 'nets'
means path is not configured well 
Fix with adding slim to path -> from directory  /models/research$ export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim


