TRAIN_PARAMETER = {
	# Cache Config Start======================
	'data_dir': 'data/MOOC1/image',
	'output_dir': 'output/MOOC1/video',
	'model_dir': 'model',
	'pretrain_dir': 'pretrain',
	# Cache Config End========================

	# Parameter Config Start==================
	'frame_interval': 5,
	'face_confidence_threshold': 1,
	'video_length_threshold': 10,    # 一般 FPS=25，这里的 总帧数=vlt*fi
	'face_size': (224, 224),
	'batch_size': 1,
	'num_workers': 0,
	'mode': 'test',
	'GPU': True,
	# Parameter Config End====================

	# Model Config Start======================
	'cur_dataset': 'video_MOOC1',
	# 'cur_dataset': 'crop_MOOC1',
	'cur_FaceEncoder': 'face_encoder',
	# Model Config End========================
}