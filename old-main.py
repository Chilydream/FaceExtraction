import os
import numpy as np
from torch.utils.data import DataLoader

from Config.TrainConfig import TRAIN_PARAMETER
from Config.DatasetConfig import DATASET_PARAMETER
from Config.NetworkConfig import NETWORK_PARAMETER
from utils import get_console_args, get_network, save_model, cosine_distance


def main():
	get_console_args()
	cur_dataset = TRAIN_PARAMETER['cur_dataset']
	face_dataset = DATASET_PARAMETER[cur_dataset]['dataset'](**DATASET_PARAMETER[cur_dataset]['d_params'])
	face_loader = DataLoader(face_dataset,
	                         batch_size=TRAIN_PARAMETER['batch_size'],
	                         num_workers=TRAIN_PARAMETER['num_workers'])
	face_encoder, _ = get_network('FaceEncoder')
	face_feature_dict = dict()
	for face_img, face_label in face_loader:
		if TRAIN_PARAMETER['GPU']:
			face_img, face_label = face_img.cuda(), face_label.cuda()
		face_feature = face_encoder(face_img)
		face_label = face_label.item()
		if face_label not in face_feature_dict.keys():
			face_feature_dict[face_label] = dict()
			face_feature_dict[face_label]['list'] = []
		if TRAIN_PARAMETER['GPU']:
			face_feature = face_feature.cpu()
		face_feature = face_feature.detach().numpy()[0]
		# magnitude = np.sqrt((face_feature**2).sum())
		magnitude = np.linalg.norm(face_feature)
		face_feature = face_feature/magnitude
		face_feature_dict[face_label]['list'].append(face_feature)

	for face_label in face_feature_dict.keys():
		if len(face_feature_dict[face_label]['list']) == 0:
			face_feature_dict.pop(face_label)
			print(face_label)
			continue
		face_feature_dict[face_label]['avg'] = np.average(face_feature_dict[face_label]['list'], axis=0)
		face_feature_dict[face_label]['var'] = np.var(face_feature_dict[face_label]['list'])

	total_face = 0
	correct_face = 0
	for face_1 in face_feature_dict.keys():
		for feature_1 in face_feature_dict[face_1]['list']:
			min_dist = None
			min_arg = None
			for face_2 in face_feature_dict.keys():
				# tmp_dist = np.linalg.norm(feature_1-face_feature_dict[face_2]['avg'])
				tmp_dist = cosine_distance(feature_1, face_feature_dict[face_2]['avg'])
				if min_dist is None or min_dist>tmp_dist:
					min_dist = tmp_dist
					min_arg = face_2
			if min_arg == face_1 and min_dist<1.5e-4:
				# print(min_dist)
				correct_face += 1
			total_face += 1

	# label2AvgFeature = dict()
	# for key in face_feature_dict.keys():
	# 	label2AvgFeature[key] = face_feature_dict[key]['avg']
	# np.save('data/MOOC1/label2AvgFeature.npy', label2AvgFeature)
	# save_model(face_encoder, 'FaceEncoder')

	print("match ratio:%.2f"%(correct_face/total_face))


if __name__ == '__main__':
	main()
