from Model import Model

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import os.path
from os import path

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
# from utils import label_map_util
# from utils import visualization_utils as vis_util
# os.chdir( '/home/yellow/models/research/object_detection' )
# import object_detection
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import ops as utils_ops

from os import listdir
from os.path import isfile, join

class ObjectDetector:
    def __init__(self):
        self.model = Model.getInstance()
        #self.model.set_name("ssd_inception_v2_coco_2018_01_28")
        tf.config.optimizer.set_jit(True)
        # What model to download.
        # Models can bee found here: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
        self.MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17'
        self.MODEL_FILE = self.MODEL_NAME + '.tar.gz'
        self.DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        self.PATH_TO_CKPT = self.MODEL_NAME + '/frozen_inference_graph.pb'

        # List of the strings that is used to add correct label for each box.
        self.PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

        # Number of classes to detect
        self.NUM_CLASSES = 90

        self.downloadModel()

    def downloadModel(self):
        print("Downloading model...")
        # Download Model
        # print("Is it File?" + str(path.isfile(self.PATH_TO_CKPT)))
        if path.isfile(self.PATH_TO_CKPT) != True:
            opener = urllib.request.URLopener()
            opener.retrieve(self.DOWNLOAD_BASE + self.MODEL_FILE, self.MODEL_FILE)
            tar_file = tarfile.open(self.MODEL_FILE)
            for file in tar_file.getmembers():
                file_name = os.path.basename(file.name)
                if 'frozen_inference_graph.pb' in file_name:
                    tar_file.extract(file, os.getcwd())

    def loadModel(self):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

    def setCustomModelSettings(self):
        self.MODEL_NAME = '/home/yellow/models/research/object_detection/inference_graph'
        self.PATH_TO_CKPT = self.MODEL_NAME + '/frozen_inference_graph.pb'  # isto?
        self.PATH_TO_LABELS = os.path.join('training', 'labelmap.pbtxt')  # !!

    def updateName(self):
        # self.model = Model.getInstance()
        self.MODEL_NAME = self.model.get_name()
        print("New value=" + self.MODEL_NAME)

    def loadLabelMap(self):
        # Loading label map
        # Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
        self.label_map = label_map_util.load_labelmap("../" + self.PATH_TO_LABELS)
        self.categories = label_map_util.convert_label_map_to_categories(
            self.label_map, max_num_classes=self.NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)

    def configureModel(self):
        self.loadModel()
        self.loadLabelMap()

    # Helper code
    def load_image_into_numpy_array(self,image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    def detectOcjectsFromCamera(self):
        print("Detecting objects from camera")
        # Define the video stream
        # self.model = Model()
        # a = self.model.bool_custom_trained
        # print(a)
        # a = self.model._Model__bool_custom_trained
        # print(a)
        if self.model.get_bool_custom_trained():
            print("CUSTOM")
            self.setCustomModelSettings()
            self.configureModel()
        else:
            self.updateName()
            self.downloadModel()
            self.configureModel()

        print(self.model.get_name())
        print(self.model.get_bool_custom_trained())
        print(self.MODEL_NAME)

        cap = cv2.VideoCapture(0)  # Change only if you have more than one webcams

        # Detection
        with self.detection_graph.as_default():
            with tf.compat.v1.Session(graph=self.detection_graph) as sess:
                while True:

                    # Read frame from camera
                    ret, image_np = cap.read()
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    # Extract image tensor
                    image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                    # Extract detection boxes
                    boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                    # Extract detection scores
                    scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                    # Extract detection classes
                    classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                    # Extract number of detectionsd
                    num_detections = self.detection_graph.get_tensor_by_name(
                        'num_detections:0')
                    # Actual detection.
                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                    # Visualization of the results of a detection.
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        self.category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8)

                    # Display output
                    cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))

                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        break

    def run_inference_for_single_image(self,image, graph):
        with graph.as_default():
            with tf.compat.v1.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                            tensor_name)
                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image.shape[1], image.shape[2])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict,
                                       feed_dict={image_tensor: image})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.int64)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict

    def detectOcjectsFromImages(self):
        print("Detecting objects from image")
        if self.model.get_bool_custom_trained():
            print("CUSTOM")
            self.setCustomModelSettings()
            self.configureModel()
        else:
            self.updateName()
            self.downloadModel()
            self.configureModel()

        self.setImageSources()
        """for image_path in self.TEST_IMAGE_PATHS:
            image = Image.open(image_path)
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = self.load_image_into_numpy_array(image)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            output_dict = self.run_inference_for_single_image(image_np_expanded, self.detection_graph)
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                self.category_index,
                instance_masks=output_dict.get('detection_masks'),
                use_normalized_coordinates=True,
                line_thickness=8)
            plt.figure(figsize=self.IMAGE_SIZE)
            plt.imshow(image_np)"""

        #return plt


    def setImageSources(self):
        print("Setting IMG sources")
        # For the sake of simplicity we will use only 2 images:
        # image1.jpg
        # image2.jpg
        # If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
        self.PATH_TO_TEST_IMAGES_DIR = 'test_images'
       # self.TEST_IMAGE_PATHS = [os.path.join(self.PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3)]
        self.TEST_IMAGE_PATHS = [join(self.PATH_TO_TEST_IMAGES_DIR, f) for f in listdir(self.PATH_TO_TEST_IMAGES_DIR) if isfile(join(self.PATH_TO_TEST_IMAGES_DIR, f))]

        for f in self.TEST_IMAGE_PATHS:
            print(f)
        # Size, in inches, of the output images.
        self.IMAGE_SIZE = (12, 8)


