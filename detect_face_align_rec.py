import torch
from yoloface_detect_align_module import yoloface
from ultraface_detect_module import ultraface
from ssdface_detect_module import ssdface
from retinaface_detect_align_module import retinaface, retinaface_dnn
from mtcnn_pfld_landmark import mtcnn_detect as mtcnnface
from facebox_detect_module import facebox_pytorch as facebox
from facebox_detect_module import facebox_dnn
from dbface_detect_align_module import dbface_detect as dbface
from centerface_detect_align_module import centerface
from lffd_detect_module import lffdface
from libfacedetect_align_module import libfacedet
from get_face_feature import arcface
from get_face_feature import arcface_dnn
import pickle
import cv2
import numpy as np
from scipy import spatial
import argparse
from PIL import Image, ImageDraw, ImageFont
import json
from  arcface.engine import *

face_engine = ArcFace()

# 需要引擎开启的功能
mask = ASF_FACE_DETECT | ASF_FACERECOGNITION 
# 初始化接口
res = face_engine.ASFInitEngine(ASF_DETECT_MODE_IMAGE,ASF_OP_0_ONLY,30,10,mask)

class Object(object):
    def __init__(self):
        self.name="webrtcHacks Yolo Face Recognition REST API"

    def toJSON(self):
        return json.dumps(self.__dict__)


def textImgUtf8(img, text, x, y):
    #cv2.putText(drawimg, pred_name, (boxs[i][0], boxs[i][1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    font = ImageFont.truetype('msyh.ttc', 16)
    pillowImage = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pillowImage)
    draw.text((x + 6, y - 20), text, font=font, fill=(255,0,0))
    #cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    frame = cv2.cvtColor(np.asarray(pillowImage),cv2.COLOR_RGB2BGR)
    return frame

def getFaces(img):
    srcimg = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    #水平翻转图片
    srcimg = cv2.flip(srcimg,1)
    #srcg = load_image_into_numpy_array(srcimg)
    #cv2.imshow('me',srcimg)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #srcimg.show()
    #rcimg = cv2.imread("D:/github/10kinds-light-face-detector-align-recognition/test/lx3.png")
    output = []
    threshold = 0.5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #face_embdnet = arcface(device=device)
    face_embdnet = arcface()
    detect_face = retinaface_dnn()
    emb_path = 'yoloface_detect_arcface_feature.pkl'
    with open(emb_path, 'rb') as f:
        dataset = pickle.load(f)
    faces_feature, names_list = dataset
    if srcimg is None:
        exit('please give correct image')
    boxes, faces_img = detect_face.get_face(srcimg)
    if len(faces_img) == 0:
        return json.dumps([ob.__dict__ for ob in output])
        exit('no detec face')
    drawimg, threshold = srcimg.copy(), 0.5
    ori_h, ori_w, _ = srcimg.shape
    print(srcimg.shape)
    #初始化返回数据对象
    item = Object()
    item.version = "0.0.1"
    item.numObjects = float(len(faces_img))
    item.threshold = threshold
    output.append(item)
    for i, face in enumerate(faces_img):
        feature_out = face_embdnet.get_feature_by_dlib_api(face)
        #print(faces_feature)
        print(feature_out)
        if len(feature_out) == 0:
            continue
        dist = spatial.distance.cdist(faces_feature, np.expand_dims(feature_out,axis=0), metric='euclidean').flatten()
        #for j in faces_feature:
        #   #res, score = face_engine.ASFFaceFeatureCompare(j, feature_out)
        #   print(score)
        #print(dist)
        min_id = np.argmin(dist)
        pred_score = 1 / (1 + dist[min_id])#欧氏距离归一化
        pred_name = 'unknow'
        #cv2.imshow("faceCrop",srcimg[boxes[i][1]:boxes[i][3],boxes[i][0]:boxes[i][2]])
        #cv2.waitKey(0)
        if pred_score >= threshold:
            pred_name = names_list[min_id]
            # Add some metadata to the output
            print(" person %s - score: %s, coordinates: %s" % (pred_name, pred_score, boxes[i]))
            x1 = boxes[i][0]
            y1 = boxes[i][1]
            x2 = boxes[i][2]
            y2 = boxes[i][3]

            item = Object()
            item.name = 'Object'
            item.class_name = pred_name
            item.score = float(pred_score)
            item.y = float(y1 / ori_h)
            item.x = float(x1 / ori_w)
            item.width = float((x2-x1)/ori_w)
            item.height = float((y2-y1)/ori_h)
            #detectEmotion(image_np[int(boxes[c][1]):int(boxes[c][1]+boxes[c][3]), int(boxes[c][0]):int(boxes[c][0]+boxes[c][2])])
            output.append(item)
    outputJson = json.dumps([ob.__dict__ for ob in output])
    return outputJson


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
    parser.add_argument('--imgpath', type=str, default='s_l.jpg', help='Path to image file.')
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    face_embdnet = arcface_dnn()
    detect_face = yoloface(device=device)
    emb_path = 'yoloface_detect_arcface_feature.pkl'
    with open(emb_path, 'rb') as f:
        dataset = pickle.load(f)
    faces_feature, names_list = dataset

    srcimg = cv2.imread(args.imgpath)
    if srcimg is None:
        exit('please give correct image')
    boxs, faces_img = detect_face.get_face(srcimg)
    if len(faces_img) == 0:
        exit('no detec face')
    drawimg, threshold = srcimg.copy(), 0.65
    for i, face in enumerate(faces_img):
        feature_out = face_embdnet.get_feature(face)
        dist = spatial.distance.cdist(faces_feature, feature_out, metric='euclidean').flatten()
        min_id = np.argmin(dist)
        pred_score = dist[min_id]
        pred_name = 'unknow'
        if dist[min_id] <= threshold:
            pred_name = names_list[min_id]
        cv2.rectangle(drawimg, (boxs[i][0], boxs[i][1]), (boxs[i][2], boxs[i][3]), (0, 0, 255), thickness=2)
        #不支持中文
        #cv2.putText(drawimg, pred_name, (boxs[i][0], boxs[i][1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        drawed = textImgUtf8(drawimg, pred_name, boxs[i][0], boxs[i][1])
    cv2.namedWindow('face recognition', cv2.WINDOW_NORMAL)
    cv2.imshow('face recognition', drawed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()