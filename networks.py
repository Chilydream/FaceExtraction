import math

import torch
import torchsnooper
from torch import nn, optim


class Reshape(nn.Module):
	def __init__(self, *args):
		super(Reshape, self).__init__()
		self.shape = args

	def forward(self, x):
		return x.view(self.shape)


# @torchsnooper.snoop()
class ResNet(nn.Module):
	def __init__(self, layers):
		super(ResNet, self).__init__()
		self.inplanes = 64
		# 输入（3, 224，224）
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
		                       bias=False)  # 这一步图片宽与高缩小为一半
		# （64, 112，112）
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 这一步图片宽与高缩小为一半
		# （64, 56，56）
		self.stack1 = self.make_stack(64, layers[0])
		# （64*4, 56，56）
		self.stack2 = self.make_stack(128, layers[1], stride=2)  # 这一步图片宽与高缩小为一半
		# （128*4, 28，28）
		self.stack3 = self.make_stack(256, layers[2], stride=2)  # 这一步图片宽与高缩小为一半
		# （256*4, 14，14）
		self.stack4 = self.make_stack(512, layers[3], stride=2)  # 这一步图片宽与高缩小为一半
		# （512*4, 7，7）
		self.avgpool = nn.AvgPool2d(7, stride=1)
		# （2048, 1，1）
		self.init_param()

	def init_param(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
				m.weight.data.normal_(0, math.sqrt(2./n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				n = m.weight.shape[0]*m.weight.shape[1]
				m.weight.data.normal_(0, math.sqrt(2./n))
				m.bias.data.zero_()

	def make_stack(self, planes, blocks, stride=1):
		downsample = None
		layers = []

		if stride != 1 or self.inplanes != planes*Bottleneck.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes*Bottleneck.expansion,
				          kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes*Bottleneck.expansion),
			)

		layers.append(Bottleneck(self.inplanes, planes, stride, downsample))
		self.inplanes = planes*Bottleneck.expansion
		for i in range(1, blocks):
			layers.append(Bottleneck(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.stack1(x)
		x = self.stack2(x)
		x = self.stack3(x)
		x = self.stack4(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		# 输出 （batchs_size, 2048）

		return x


class Bottleneck(nn.Module):
	expansion = 4

	# 如果 stride=1，最后输出（w, h, planes*4）
	# 如果 stride=2，最后输出（w/2, h/2, planes*4）
	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(planes*4)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out


# @torchsnooper.snoop()
class FaceEncoder(nn.Module):
	def __init__(self, layers):
		super(FaceEncoder, self).__init__()
		self.model = nn.Sequential(
			ResNet(layers),
			nn.Linear(2048, 128, bias=False)
		)

	def forward(self, x):
		return self.model(x)
