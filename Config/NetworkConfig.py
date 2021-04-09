from torch import optim

from networks import FaceEncoder


NETWORK_PARAMETER = {
	'FaceEncoder': {
		'network': FaceEncoder,
		'n_params': {
			'layers': [3, 4, 6, 3],
		},
		'model_path': 'FaceEncoder.pkl',
		'optim': optim.Adam,
		'o_params': {
			'lr': 0.001,
		},
	},
}
