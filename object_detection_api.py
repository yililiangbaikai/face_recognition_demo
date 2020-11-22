# IMPORTS

import numpy as np
import os
import six.moves.urllib as urllib
import tarfile
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior
import json
from keras.models import load_model
import cv2
from PIL import Image

#if tf.__version__ != '1.4.0':
#  raise ImportError('Please upgrade your tensorflow installation to v1.4.0!')

# ENV SETUP  ### CWH: remove matplot display and manually add paths to references

# Object detection imports
from object_detection.utils import label_map_util    ### CWH: Add object_detection path

# Model Preparation

# What model to download.
#MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
#MODEL_FILE = MODEL_NAME + '.tar.gz'
#DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './models/frozen_inference_graph_face.pb'

# List of the strings that is used to add correct label for each box.
#PATH_TO_LABELS = os.path.join('object_detection/data', 'mscoco_label_map.pbtxt') ### CWH: Add object_detection path
PATH_TO_LABELS = './protos/face_label_map.pbtxt'

#NUM_CLASSES = 90
#两个类别face和background
NUM_CLASSES = 2


# Download Model
#opener = urllib.request.URLopener()
#opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
#tar_file = tarfile.open(MODEL_FILE)
#for file in tar_file.getmembers():
#  file_name = os.path.basename(file.name)
#  if 'frozen_inference_graph.pb' in file_name:
#    tar_file.extract(file, os.getcwd())


# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

config = tf.ConfigProto()#对session进行参数配置
config.allow_soft_placement=True #如果你指定的设备不存在，允许TF自动分配设备
config.gpu_options.per_process_gpu_memory_fraction=0.8#分配百分之七十的显存给程序使用，避免内存溢出，可以自己调整
config.gpu_options.allow_growth = True#按需分配显存，这个比较重要
sess = tf.Session(config=config,graph=detection_graph)
#with detection_graph.as_default():
  #  with tf.Session(graph=detection_graph) as sess:
# Definite input and output Tensors for detection_graph
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
# Each box represents a part of the image where a particular object was detected.
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
# Each score represent how level of confidence for each of the objects.
# Score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')
# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
print(categories)
category_index = label_map_util.create_category_index(categories)


# Helper code
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# added to put object in JSON
class Object(object):
    def __init__(self):
        self.name="webrtcHacks TensorFlow Object Detection REST API"

    def toJSON(self):
        return json.dumps(self.__dict__)

model = load_model("./model_v6_23.hdf5")
emotion_dict= {'生气': 0, '悲伤': 5, '中性': 4, '厌恶': 1, '惊讶': 6, '恐惧': 2, '高兴': 3}
def detectEmotion(image_np):
    img = 255 * image_np.astype('uint8')
    img = Image.fromarray(np.array(img), 'RGB')
    cv2.imshow('test', img)
    cv2.waitKey(0)
    #greyFrame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #faceCrop = greyFrame[y:y+h,x:x+w]
    faceCrop = cv2.resize(image, (48,48))
    face_image = np.reshape(faceCrop, [1, faceCrop.shape[0], faceCrop.shape[1], 1])
    predicted_class = np.argmax(model.predict(face_image))
    # 分类情绪
    label_map = dict((v,k) for k,v in emotion_dict.items()) 
    name = label_map[predicted_class]
    print(name)
    return name

def get_objects(image, threshold=0.5):
    image_np = load_image_into_numpy_array(image)

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.   
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    classes = np.squeeze(classes).astype(np.int32)
    print(classes)
    scores = np.squeeze(scores)
    boxes = np.squeeze(boxes)

    obj_above_thresh = sum(n > threshold for n in scores)
    print("detected %s objects in image above a %s score" % (obj_above_thresh, threshold))

    output = []

    # Add some metadata to the output
    item = Object()
    item.version = "0.0.1"
    item.numObjects = float(obj_above_thresh)
    item.threshold = threshold
    output.append(item)

    for c in range(0, len(category_index)):
        class_name = 'face'#category_index[c]['name']
        if scores[c] >= threshold:      # only return confidences equal or greater than the threshold
            print(" object %s - score: %s, coordinates: %s" % (class_name, scores[c], boxes[c]))

            item = Object()
            item.name = 'Object'
            item.class_name = class_name
            item.score = float(scores[c])
            item.y = float(boxes[c][0])
            item.x = float(boxes[c][1])
            item.height = float(boxes[c][2])
            item.width = float(boxes[c][3])
            #detectEmotion(image_np[int(boxes[c][1]):int(boxes[c][1]+boxes[c][3]), int(boxes[c][0]):int(boxes[c][0]+boxes[c][2])])
            output.append(item)

    outputJson = json.dumps([ob.__dict__ for ob in output])
    return outputJson