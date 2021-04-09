import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import imghdr
import cv2
import PIL
from Config.TrainConfig import TRAIN_PARAMETER
from Config.DatasetConfig import DATASET_PARAMETER
from mtcnn import mtcnn
from torch_mtcnn import detect_faces

image = cv2.imread("1.png")
detector = mtcnn.MTCNN()
info_list = detector.detect_faces(image)
bbox = info_list[0]['box']
face = image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
cv2.imwrite("face.png", face)



# people_list = os.listdir(TRAIN_PARAMETER['data_dir'])
# print(people_list)
# video_list = []
# video_id_list = []
# for ipeople, people in enumerate(people_list):
# 	video_dir = os.path.join(TRAIN_PARAMETER['data_dir'], people, 'videos')
# 	if not os.path.exists(video_dir):
# 		continue
# 	video_filelist = os.listdir(video_dir)
# 	for video_file in video_filelist:
# 		video_path = os.path.join(video_dir, video_file)
# 		if not os.path.exists(video_path):
# 			continue
# 		video_capture = cv2.VideoCapture(video_path)
# 		fps = video_capture.get(cv2.CAP_PROP_FPS)
# 		if fps==0 or video_capture is None:
# 			continue
# 		print(fps)
# 		video_list.append(os.path.join(people, 'videos', video_file))
# 		video_id_list.append(ipeople)
# video_ndarray = np.array(video_list)
# video_id_ndarry = np.array(video_id_list)
# np.save("data/MOOC1/video_filename.npy", video_ndarray)
# np.save("data/MOOC1/video_identity.npy", video_id_ndarry)

# face_raw_list = []
# face_raw_id_list = []
# for ipeople, people in enumerate(people_list):
# 	img_dir = os.path.join(TRAIN_PARAMETER['data_dir'], people, 'face_gt')
# 	if not os.path.exists(img_dir):
# 		continue
# 	img_filelist = os.listdir(img_dir)
# 	for img_file in img_filelist:
# 		if imghdr.what(os.path.join(TRAIN_PARAMETER['data_dir'], people, 'face_gt', img_file)) is None:
# 			continue
# 		face_raw_list.append(os.path.join(people, 'face_gt', img_file))
# 		face_raw_id_list.append(ipeople)
# face_raw_ndarray = np.array(face_raw_list)
# face_raw_id_ndarray = np.array(face_raw_id_list)
# np.save("data/MOOC1/face_raw_filename.npy", face_raw_ndarray)
# np.save("data/MOOC1/face_raw_identity.npy", face_raw_id_ndarray)

# a = np.load('data/MOOC1/label2AvgFeature.npy', allow_pickle=True)
# print(type(a))
# a = a.item()
# print(type(a))
# print(len(a.keys()))
