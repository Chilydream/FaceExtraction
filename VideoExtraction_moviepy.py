import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
	tf.config.experimental.set_memory_growth(gpu, True)

import cv2
import mtcnn
import torch
import torchsnooper
import moviepy.editor as mpe
from torch.utils.data import DataLoader

from Config.TrainConfig import TRAIN_PARAMETER
from Config.DatasetConfig import DATASET_PARAMETER
from Config.NetworkConfig import NETWORK_PARAMETER
from utils import get_console_args, get_network, save_model, cosine_distance, make_image_square


class FaceDetector:
	def __init__(self):
		self.face_encoder, _ = get_network('FaceEncoder', 'test')
		self.detector = mtcnn.MTCNN()
		self.image_cnt = 0

	@staticmethod
	# @torchsnooper.snoop()
	def pre_process(face_img):
		face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
		# face_img = (face_img-127.5)/127.5
		face_img = make_image_square(face_img)
		face_img = cv2.resize(face_img, TRAIN_PARAMETER['face_size'])
		face_img = np.transpose(face_img, (2, 0, 1))
		face_img = face_img.astype(np.float32)
		face_img = torch.tensor(face_img)
		face_img = face_img.unsqueeze(0)
		if TRAIN_PARAMETER['GPU']:
			face_img = face_img.cuda()
		return face_img

	# @torchsnooper.snoop()
	def detect_face(self, frame_list, gt_feature):
		exist_face = 0
		for frame in frame_list:
			info_list = self.detector.detect_faces(frame)
			for info_box in info_list:
				if info_box['confidence']<0.9:
					continue
				bbox = info_box['box']
				face_img = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
				self.image_cnt += 1
				face_img = FaceDetector.pre_process(face_img)
				face_feature = self.face_encoder(face_img)
				face_feature = face_feature.squeeze()
				if TRAIN_PARAMETER['GPU']:
					face_feature = face_feature.cpu()
				face_feature = face_feature.detach().numpy()
				face_dist = cosine_distance(face_feature, gt_feature)
				if face_dist<1e-4:
					exist_face += 1
					break
				elif face_dist<1.5e-4:
					exist_face += 0.5
					break
		return exist_face


def main():
	get_console_args()
	cur_dataset = TRAIN_PARAMETER['cur_dataset']
	frame_interval = TRAIN_PARAMETER['frame_interval']
	dataset = DATASET_PARAMETER[cur_dataset]['dataset'](**DATASET_PARAMETER[cur_dataset]['d_params'])
	dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

	label2AvgFeature = np.load('data/MOOC1/label2AvgFeature.npy', allow_pickle=True).item()
	detector = FaceDetector()
	output_video_cnt = 0
	skip_flag = 0
	for video_filename, label in dataloader:
		print('Start processing file:\n%s'%video_filename)
		if skip_flag<=14:
			skip_flag += 1
			continue
		video_filename, label = video_filename[0], label[0]
		if TRAIN_PARAMETER['GPU']:
			label = label.cpu()
		label = label.item()
		output_video_path = os.path.join(TRAIN_PARAMETER['output_dir'], str(label))
		# if os.path.exists(output_video_path):
		# 	continue
		gt_feature = label2AvgFeature[label]

		video_capture = cv2.VideoCapture(video_filename)
		confidence_list = []
		frame_cnt = 0
		subset_flag = 0
		debug_img_cnt = 0
		while True:
			success, frame = video_capture.read()
			if not success:
				break
			frame_cnt += 1
			if frame is None or len(frame.shape)<3:
				print('ERROR1:', frame_cnt)
				subset_flag = -9
			elif subset_flag<TRAIN_PARAMETER['face_confidence_threshold']:
				try:
					face_confidence = detector.detect_face([frame], gt_feature)
				except:
					# cv2.imwrite('debug/image/'+str(debug_img_cnt)+'.png', frame)
					# debug_img_cnt += 1
					print('ERROR2:', frame_cnt)
					face_confidence = -999
				subset_flag += face_confidence
			if frame_cnt%frame_interval==0:
				confidence_list.append(subset_flag)
				subset_flag = 0

		print('\tStart cliping video')
		start_pos = None
		output_fps = 25
		if not os.path.exists(output_video_path):
			os.mkdir(output_video_path)
		output_video_path = output_video_path+'/%d.mp4'
		with mpe.VideoFileClip(video_filename) as video:
			for i in range(len(confidence_list)):
				if confidence_list[i]>=TRAIN_PARAMETER['face_confidence_threshold']:
					if start_pos is None:
						start_pos = i
				else:
					if start_pos is not None:
						tmp_length = i-start_pos
						if tmp_length>TRAIN_PARAMETER['video_length_threshold']:
							print('\tClip %d video now'%output_video_cnt)
							sub_speech = video.subclip(start_pos*frame_interval/output_fps, i*frame_interval/output_fps)
							sub_speech.write_videofile(output_video_path%output_video_cnt, fps=output_fps)
							output_video_cnt += 1
							start_pos = None


if __name__ == '__main__':
	main()
