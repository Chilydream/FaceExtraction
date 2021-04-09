import argparse
import os
import numpy as np
import cv2
import torch
from torch import optim

from Config.TrainConfig import TRAIN_PARAMETER
from Config.NetworkConfig import NETWORK_PARAMETER


def get_console_args():
	parser = argparse.ArgumentParser(description='Face Extraction')
	parser.add_argument('-m', '--mode', choices=['train', 'continue', 'test'], default=TRAIN_PARAMETER['mode'])
	args = parser.parse_args()
	TRAIN_PARAMETER['mode'] = args.mode


# 输入RGB格式的图片，返回填充为正方形的图片
def make_image_square(img):
	s = max(img.shape[0:2])
	f = np.zeros((s, s, 3), np.uint8)
	ax, ay = (s-img.shape[1])//2, (s-img.shape[0])//2
	f[ay:img.shape[0]+ay, ax:ax+img.shape[1]] = img
	return f


def random_horizon_flip(img):
	if np.random.uniform()<0.5:
		img = cv2.flip(img, 1)
	return img


def cosine_distance(x, y):
	x_norm = np.linalg.norm(x)
	y_norm = np.linalg.norm(y)
	if x_norm*y_norm==0:
		similiarity = 0
		print(x, y)
	else:
		similiarity = np.dot(x, y.T)/(x_norm*y_norm)
	dist = 1-similiarity
	return dist


def feature_unitization(x):
	magnitude = np.linalg.norm(x)
	x = x/magnitude
	return x


class Meter(object):
	def __init__(self, name, display, fmt=':f'):
		self.name = name
		self.display = display
		self.fmt = fmt
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val*n
		self.count += n
		self.avg = self.sum/self.count

	def __str__(self):
		fmtstr = '{name}:{'+self.display+self.fmt+'},'
		return fmtstr.format(**self.__dict__)


def get_network(net_name, mode=TRAIN_PARAMETER['mode']):
	net_params = NETWORK_PARAMETER[net_name]
	network = net_params['network'](**net_params['n_params'])
	if TRAIN_PARAMETER['GPU']:
		network = network.cuda()

	optimizer = None
	if mode == 'train':
		network.train()
		optimizer = net_params['optim'](network.parameters(), **net_params['o_params'])
	elif mode == 'continue':
		network.train()
		pretrain_model = os.path.join(TRAIN_PARAMETER['pretrain_dir'], net_params['model_path'])
		if TRAIN_PARAMETER['GPU']:
			network.load_state_dict(torch.load(pretrain_model), strict=False)
		else:
			network.load_state_dict(torch.load(pretrain_model, map_location='cpu'), strict=False)
		optimizer = net_params['optim'](network.parameters(), **net_params['o_params'])
	elif mode == 'test' or mode == 'dev':
		network.eval()
		pretrain_model = os.path.join(TRAIN_PARAMETER['pretrain_dir'], net_params['model_path'])
		if TRAIN_PARAMETER['GPU']:
			network.load_state_dict(torch.load(pretrain_model), strict=False)
		else:
			network.load_state_dict(torch.load(pretrain_model, map_location='cpu'), strict=False)

	return network, optimizer


def cycle(dataloader):
	while True:
		for data in dataloader:
			yield data


def save_model(network, model_name):
	if not os.path.exists(TRAIN_PARAMETER['model_dir']):
		os.makedirs(TRAIN_PARAMETER['model_dir'])

	if model_name in NETWORK_PARAMETER.keys():
		model_path = os.path.join(TRAIN_PARAMETER['model_dir'], NETWORK_PARAMETER[model_name]['model_path'])
	else:
		if not model_name.endswith('.pkl') or not model_name.endswith('.pth'):
			model_name = model_name+'.pkl'
		model_path = os.path.join(TRAIN_PARAMETER['model_dir'], model_name)
		print('WARNING: NETWORK_PARAMETER中没有该网络参数，模型已保存至以下地址\n%s'%model_path)
	torch.save(network.state_dict(), model_path)
