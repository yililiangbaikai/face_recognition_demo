import cv2
from arcface.resnet import resnet_face18
from facebox_detect_module import facebox_pytorch
import torch
import numpy as np
import os
import pickle
import sys
from collections import OrderedDict
from  arcface.engine import *
from face_recognition.face_recognition import *
import dlib

APPID = b'8JXPmd4M9CG4ZVi4nejVio7VfJszvFab9idJGoepFfCt'
SDKKey = b'8nW71QgkDWDhoQRcBJocRTTrH6Pb2t5DjE9KStKB8voT'

#激活接口,首次需联网激活ArcFace
res = ASFOnlineActivation(APPID, SDKKey)
if (MOK != res and MERR_ASF_ALREADY_ACTIVATED != res):
    print("ASFActivation fail: {}".format(res))
else:
    print("ASFActivation sucess: {}".format(res))
#初始化ArcFace
face_engine = ArcFace()
# 需要引擎开启的功能
mask = ASF_FACE_DETECT | ASF_FACERECOGNITION | ASF_FACE3DANGLE
# 初始化接口
res = face_engine.ASFInitEngine(ASF_DETECT_MODE_IMAGE,  ASF_OP_0_ONLY, 32, 10, mask)
if (res != MOK):
    print("ASFInitEngine fail: {}".format(res) )
else:
    print("ASFInitEngine sucess: {}".format(res))

def convert_onnx():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = 'arcface/resnet18_110.pth'
    model = resnet_face18(use_se=False)
    # model = torch.nn.DataParallel(model)
    # model.load_state_dict(torch.load(model_path, map_location=device))

    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k.replace('module.', '')] = v    ## remove 'module.'
    model.load_state_dict(new_state_dict)

    model.to(device)
    model.eval()
    dummy_input = torch.randn(1, 1, 128, 128).to(device)
    onnx_path = 'arcface/resnet18_110.onnx'
    torch.onnx.export(model, dummy_input, onnx_path, input_names=['input'], output_names=['output'])

class arcface():
    def __init__(self, model_path='arcface/resnet18_110.pth', device = 'cuda'):
        self.model = resnet_face18(use_se=False)
        # self.model = torch.nn.DataParallel(self.model)
        # self.model.load_state_dict(torch.load(model_path, map_location=device))

        state_dict = torch.load(model_path, map_location=device)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace('module.', '')] = v  ## remove 'module.'
        self.model.load_state_dict(new_state_dict)

        self.model.to(device)
        self.model.eval()
        self.device = device
    def get_feature(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        img = np.transpose(img, axes=(2,0,1))
        img = img[np.newaxis, np.newaxis, :, :]
        # img = img[np.newaxis, :, :, :]
        img = img.astype(np.float32, copy=False)
        img -= 127.5
        img /= 127.5
        with torch.no_grad():
            data = torch.from_numpy(img).to(self.device)
            output = self.model(data)
            output = output.data.cpu().numpy()
        return output
    def get_feature_by_arcface_cppapi(self, srcimg):

        #检测第一张图中的人脸
        res,detectedFaces = face_engine.ASFDetectFaces(srcimg)
        face_feature = object()
        print(detectedFaces)
        if res == MOK:
            single_detected_face = ASF_SingleFaceInfo()
            single_detected_face.faceRect = detectedFaces.faceRect[0]
            single_detected_face.faceOrient = detectedFaces.faceOrient[0]
            print(single_detected_face)
            res ,face_feature= face_engine.ASFFaceFeatureExtract(srcimg,single_detected_face)
            #print(face_feature1)
            #print(face_feature1.feature)
            #print(face_feature1.feature.decode('utf-8'))
            #print(face_feature1.featureSize)
            if (res != MOK):
                print ("ASFFaceFeatureExtract fail: {}".format(res))
        else:
            print("ASFDetectFaces fail: {}".format(res))
        return face_feature

    def get_feature_by_dlib_api(self, srcimg):
        #cv2.imshow("detect_face_to_feature", srcimg)
        #cv2.waitKey(0)
        sicp = srcimg.copy()
        sicpCv = cv2.cvtColor(sicp, cv2.COLOR_BGR2RGB)
        encodings = api.face_encodings(np.asarray(sicpCv))
        if(len(encodings) > 0):
            print(encodings)
            return encodings[0]
        else:
            print("no feature found!" + str(encodings))
        return []

class arcface_dnn():
    def __init__(self, model_path='arcface/resnet18_110.onnx'):
        self.model = cv2.dnn.readNetFromONNX(model_path)
        self.input_size = (128, 128)
    def get_feature(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, self.input_size, interpolation=cv2.INTER_AREA)
        blob = cv2.dnn.blobFromImage(img, scalefactor=1 / 127.5, mean=127.5)
        self.model.setInput(blob)
        output = self.model.forward(['output'])
        return output[0]




if __name__ == '__main__':
    #from yoloface_detect_align_module import yoloface    ###你还可以选择其他的人脸检测器
    from retinaface_detect_align_module import retinaface_dnn
    from ultraface_detect_module import ultraface

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # face_embdnet = arcface(device=device)
    face_embdnet = arcface()   ###已调试通过，与pytorch版本的输出结果吻合
    detect_face = retinaface_dnn()

    out_emb_path = 'yoloface_detect_arcface_feature.pkl'
    imgroot = 'D:/tfObjWebrtc/image'
    dirlist = os.listdir(imgroot)    ### imgroot里有多个文件夹，每个文件夹存放着一个人物的多个肖像照，文件夹名称是人名
    feature_list, name_list = [], []
    for i,name in enumerate(dirlist):
        sys.stdout.write("\rRun person{0}, name:{1}".format(i, name))
        sys.stdout.flush()

        imgdir = os.path.join(imgroot, name)
        #跳过普通文件只处理文件夹
        if os.path.isfile(imgdir):
            continue;
        imglist = os.listdir(imgdir)
        for imgname in imglist:
            #srcimg = cv2.imread(os.path.join(imgdir, imgname))
            #支持中文读取
            
            srcimg = cv2.imdecode(np.fromfile(os.path.join(imgdir, imgname), dtype=np.uint8), 1)
            _, face_img = detect_face.detect(srcimg)  ###肖像照，图片中有且仅有有一个人脸
            if len(face_img) > 1:
                continue
            feature_out = face_embdnet.get_feature_by_dlib_api(face_img[0])
            #保存图片支持中文cv2.imencode('.jpg', src)[1].tofile(save_path)
            #_, face_img = detect_face.detect(srcimg)  ###肖像照，图片中有且仅有有一个人脸
            #图片裁剪 必须是4的倍数
            #height, width = srcimg.shape[:2]
            #srcimg = srcimg[:, :width - width%4]
            #cv2.imshow("crop", srcimg)
            #cv2.waitKey(0)
            #feature_out = face_embdnet.get_feature_by_arcface_cppapi(srcimg)
            #if len(face_img)!=1:
            #    print(imgname)
            #    continue
            #cv2.imshow("crop", srcimg)
            #cv2.waitKey(0)
            #feature_out = face_embdnet.get_feature(face_img[0])
            #cv2.imshow("get_feature",face_img[0])
            #cv2.waitKey(0)
          
            if len(feature_out) > 0:
                feature_list.append(feature_out.tolist())
                name_list.append(name)
                #feature_out = feature_out[0]

            print(feature_out) 
            print(type(feature_out))
            #print(feature_out.tolist())
            #print(type(feature_out.tolist()))
            #print(np.asarray(feature_out))
            #print(type(np.asarray(feature_out)))
    print(name_list)
    print(feature_list)
    print(np.asarray(feature_list))
    print(np.squeeze(np.asarray(feature_list)))
    #face_feature = (np.squeeze(np.asarray(feature_list)), name_list)
    face_feature = (feature_list, name_list)

    with open(out_emb_path, 'wb') as f:
        pickle.dump(face_feature, f)