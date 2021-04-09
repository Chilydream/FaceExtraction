from Config.TrainConfig import TRAIN_PARAMETER
from dataset import CACD2000_img_dataset, MOOC_img_dataset, MOOC_video_dataset

DATASET_PARAMETER = {
	'CACD2000': {
		'dataset': CACD2000_img_dataset,
		'd_params': {
			'root_path': "data/CACD2000/image/",
			'label_path': "data/CACD2000/label.npy",
			'name_path': "data/CACD2000/name.npy",
			'train_mode': TRAIN_PARAMETER['mode'],
		},
	},
	'raw_MOOC1': {
		'dataset': MOOC_img_dataset,
		'd_params': {
			'root_dir': "data/MOOC1/image",
			'path_metadata': 'data/MOOC1/face_raw_filename.npy',
			'label_metadata': 'data/MOOC1/face_raw_identity.npy',
		},
	},
	'crop_MOOC1': {
		'dataset': MOOC_img_dataset,
		'd_params': {
			'root_dir': "data/MOOC1/image",
			'path_metadata': 'data/MOOC1/face_crop_filename.npy',
			'label_metadata': 'data/MOOC1/face_crop_identity.npy',
		},
	},
	'video_MOOC1': {
		'dataset': MOOC_video_dataset,
		'd_params': {
			'root_dir': "data/MOOC1/image",
			'path_metadata': 'data/MOOC1/video_filename.npy',
			'label_metadata': 'data/MOOC1/video_identity.npy',
		}
	},
}
