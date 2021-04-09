import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from Config.TrainConfig import TRAIN_PARAMETER
from utils import make_image_square, random_horizon_flip


class MOOC_img_dataset(Dataset):
	def __init__(self, root_dir, path_metadata, label_metadata, train_mode=TRAIN_PARAMETER['mode']):
		self.root_dir = root_dir
		self.image_list, self.label_list = [], []

		image_filenames = np.load(path_metadata)
		image_labels = np.load(label_metadata)
		for filename, label in zip(image_filenames, image_labels):
			img_filename = os.path.join(self.root_dir, filename)
			img = cv2.imread(img_filename)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			img = make_image_square(img)
			img = cv2.resize(img, TRAIN_PARAMETER['face_size'])
			img = np.transpose(img, (2, 0, 1))
			img = (img-127.5)/127.5     # ques:之前好像没加这句
			if train_mode == 'train':
				img = random_horizon_flip(img)
			img = img.astype(np.float32)
			self.image_list.append(img)
			self.label_list.append(label)

	def __getitem__(self, index):
		return self.image_list[index], self.label_list[index]

	def __len__(self):
		return len(self.label_list)


class MOOC_video_dataset(Dataset):
	def __init__(self, root_dir, path_metadata, label_metadata, train_mode=TRAIN_PARAMETER['mode']):
		self.root_dir = root_dir
		self.video_list = []
		self.label_list = []

		video_filenames = np.load(path_metadata)
		video_label = np.load(label_metadata)
		for filename, label in zip(video_filenames, video_label):
			video_filename = os.path.join(root_dir, filename)
			# video_capture = cv2.VideoCapture(video_filename)
			# self.video_list.append(video_capture)
			self.video_list.append(video_filename)
			self.label_list.append(label)

	def __getitem__(self, item):
		return self.video_list[item], self.label_list[item]

	def __len__(self):
		return len(self.video_list)


class CACD2000_img_dataset(Dataset):
	def __init__(self, root_path, label_path, name_path, train_mode=TRAIN_PARAMETER['mode']):
		"""
		Initialize some variables
		Load labels & names
		define transform
		"""
		self.root_path = root_path
		self.image_labels = np.load(label_path)
		self.image_names = np.load(name_path)
		self.train_mode = train_mode
		self.transform = {
			'train': transforms.Compose([
				transforms.Resize(224),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),  # 自带归一化，会将图片归一化到 [0, 1]
				#               transforms.Normalize([0.656,0.487,0.411], [1., 1., 1.])
			]),
			'val': transforms.Compose([
				transforms.Resize(224),
				transforms.ToTensor(),
				#               transforms.Normalize([0.656,0.487,0.411], [1., 1., 1.])
			]),
		}

	def __len__(self):
		"""
		Get the length of the entire dataset
		"""
		print("Length of dataset is ", self.image_labels.shape[0])
		return self.image_labels.shape[0]

	def __getitem__(self, idx):
		"""
		Get the image item by index
		"""
		image_name = os.path.join(self.root_path, self.image_names[idx])
		image = Image.open(image_name)
		image_label = self.image_labels[idx].astype(int)-1
		transformed_img = self.transform[self.train_mode](image)
		sample = {'image': transformed_img, 'label': torch.from_numpy(image_label)}
		return sample
