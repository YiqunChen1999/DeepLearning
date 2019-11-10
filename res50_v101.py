"""
	@Author		Yiqun Chen
	@Time		2019-11-09
	@Info		res50 for cifar-10, using PyTorch
	@Modified
"""

# import the main packages.
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/res50/v101')

# import auxiliary packages.
import os
import time

# define the constants.
INFO = '****>>>>'
BATCH_SIZE = 128

'''
	@Input		None
	@Output		training_data_loader: loading the training data; 
				testing_data_loader: loading the testing data;
				classes: the classes of the cifar-10
	@Usage		
		training_data_loader, testing_data_loader, classes = get_data()
'''
def get_data():
	print(INFO, 'loading the training and testing data....')

	transform = transforms.Compose([
		transforms.ToTensor(), 
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	# loading the training data.
	training_set = torchvision.datasets.CIFAR10(
		root='../data', 
		train=True, 
		download=False, 
		transform=transform)
	training_data_loader = torch.utils.data.DataLoader(
		training_set, 
		batch_size=BATCH_SIZE, 
		shuffle=True, 
		num_workers=2)

	# loading the testing data.
	testing_set = torchvision.datasets.CIFAR10(
		root='../data', 
		train=False, 
		download=False, 
		transform=transform)
	testing_data_loader = torch.utils.data.DataLoader(
		testing_set, 
		batch_size=BATCH_SIZE, 
		shuffle=False, 
		num_workers=2)

	classes = ('plane', 'car', 'bird', 'cat', 'deer',
	 'dog', 'frog', 'horse', 'ship', 'truck')

	print(INFO, 'done!')
	return training_data_loader, testing_data_loader, classes


def set_params(model, training_data_loader, testing_data_loader):
	print(INFO, 'setting params...')
	loss_fn = nn.CrossEntropyLoss()
	training_params = {
	'epochs': 64000, 
	'data': training_data_loader, 
	'device': torch.device('cuda:3'),
	'optimizer': torch.optim.SGD(
		model.parameters(),
		lr=0.01, 
		momentum=0.9, 
		weight_decay=0.0001),
	'loss_fn': loss_fn
	}
	evaluating_params = {
	'data': testing_data_loader,
	'device': torch.device('cuda:3'),
	'loss_fn': loss_fn,
	}
	print(INFO, 'done!')
	return training_params, evaluating_params


def set_res50_params():
	print(INFO, 'setting parameters of Res50...')
	
	conv_2_1_params = {
	'input_channels': 256, 
	'output_channels': 64
	}
	conv_2_2_params = {
	'input_channels': 64, 
	'output_channels': 64
	}
	conv_2_3_params = {
	'input_channels': 64, 
	'output_channels': 256
	}
	conv2_x_params = {
	'conv1': conv_2_1_params, 
	'conv2': conv_2_2_params, 
	'conv3': conv_2_3_params
	}

	conv_3_1_params = {
	'input_channels': 512, 
	'output_channels': 128
	}
	conv_3_2_params = {
	'input_channels': 128, 
	'output_channels': 128
	}
	conv_3_3_params = {
	'input_channels': 128, 
	'output_channels': 512
	}
	conv3_x_params = {
	'conv1': conv_3_1_params, 
	'conv2': conv_3_2_params, 
	'conv3': conv_3_3_params
	}

	conv_4_1_params = {
	'input_channels': 1024, 
	'output_channels': 256
	}
	conv_4_2_params = {
	'input_channels': 256, 
	'output_channels': 256
	}
	conv_4_3_params = {
	'input_channels': 256, 
	'output_channels': 1024
	}
	conv4_x_params = {
	'conv1': conv_4_1_params, 
	'conv2': conv_4_2_params, 
	'conv3': conv_4_3_params
	}

	conv_5_1_params = {
	'input_channels': 2048, 
	'output_channels': 512
	}
	conv_5_2_params = {
	'input_channels': 512, 
	'output_channels': 512
	}
	conv_5_3_params = {
	'input_channels': 512, 
	'output_channels': 2048
	}
	conv5_x_params = {
	'conv1': conv_5_1_params, 
	'conv2': conv_5_2_params, 
	'conv3': conv_5_3_params
	}

	res50_params = {
	'conv2_x': conv2_x_params, 
	'conv3_x': conv3_x_params, 
	'conv4_x': conv4_x_params, 
	'conv5_x': conv5_x_params
	}

	return res50_params

'''

'''
class ResBlock(nn.Module):
	def __init__(self, params, size_down=False):
		super(ResBlock, self).__init__()
		self.params = params
		self.size_down = size_down
		if self.size_down:
			conv1_stride = 2
		else:
			conv1_stride = 1
		self.conv1 = nn.Conv2d(
			in_channels=self.params['conv1']['input_channels'], 
			out_channels=self.params['conv1']['output_channels'], 
			kernel_size=1, 
			stride=conv1_stride, 
			padding=0)
		self.conv2 = nn.Conv2d(
			in_channels=self.params['conv2']['input_channels'], 
			out_channels=self.params['conv2']['output_channels'], 
			kernel_size=3, 
			padding=1)
		self.conv3 = nn.Conv2d(
			in_channels=self.params['conv3']['input_channels'], 
			out_channels=self.params['conv3']['output_channels'], 
			kernel_size=1, 
			padding=0)
		self.conv_proj = nn.Conv2d(
			in_channels=self.params['conv1']['input_channels'], 
			out_channels=self.params['conv3']['output_channels'], 
			kernel_size=1, 
			stride=conv1_stride,
			padding=0)
		self.norm1 = nn.BatchNorm2d(
			num_features=self.params['conv1']['output_channels'])
		self.norm2 = nn.BatchNorm2d(
			num_features=self.params['conv2']['output_channels'])
		self.norm3 = nn.BatchNorm2d(
			num_features=self.params['conv3']['output_channels'])
		self.norm_proj = nn.BatchNorm2d(
			num_features=self.params['conv3']['output_channels'])

	def forward(self, x):
		#print(INFO, x.shape)
		identity_x = self.conv_proj(x)
		#print(INFO, x.shape)
		identity_x = self.norm_proj(identity_x)
		#print(INFO, x.shape)
		x = self.conv1(x)
		#print(INFO, x.shape)
		x = self.norm1(x)
		#print(INFO, x.shape)
		x = F.relu(x)
		#print(INFO, x.shape)
		x = self.conv2(x)
		x = self.norm2(x)
		x = F.relu(x)
		x = self.conv3(x)
		x = self.norm3(x)
		x = F.relu(x)
		x = x + identity_x
		x = F.relu(x)
		return x

class Res50(nn.Module):
	def __init__(self, res50_params):
		super(Res50, self).__init__()
		print(INFO, 'constructing the Res50...')
		self.res50_params = res50_params
		
		# construct the conv1
		self.conv1_x = nn.Conv2d(
			in_channels=3, 
			out_channels=64, 
			kernel_size=1)
		self.norm1 = nn.BatchNorm2d(
			num_features=64)

		# construct the conv2_x
		self.conv2_x_params = self.res50_params['conv2_x']
		# self.max_pooling = nn.MaxPool2d(
		# 	kernel_size=3,
		# 	padding=1, 
		# 	stride=2)
		first_block_params = res50_params['conv2_x']
		first_block_params['conv1']['input_channels'] = 64
		self.conv2_x_1 = ResBlock(
			params=first_block_params,
			size_down=False)
		first_block_params['conv1']['input_channels'] = 256
		self.conv2_x_2 = ResBlock(
			params=self.conv2_x_params, 
			size_down=False)
		self.conv2_x_3 = ResBlock(
			params=self.conv2_x_params, 
			size_down=True)

		# construct the conv3_x
		self.conv3_x_params = self.res50_params['conv3_x']
		first_block_params = res50_params['conv3_x']
		first_block_params['conv1']['input_channels'] = 256
		self.conv3_x_1 = ResBlock(
			params=first_block_params, 
			size_down=False)
		first_block_params['conv1']['input_channels'] = 512
		self.conv3_x_2 = ResBlock(
			params=self.conv3_x_params, 
			size_down=False)
		self.conv3_x_3 = ResBlock(
			params=self.conv3_x_params, 
			size_down=False)
		self.conv3_x_4 = ResBlock(
			params=self.conv3_x_params,
			size_down=True)

		# construct the conv4_x
		self.conv4_x_params = self.res50_params['conv4_x']
		first_block_params = res50_params['conv4_x']
		first_block_params['conv1']['input_channels'] = 512
		self.conv4_x_1 = ResBlock(
			params=first_block_params, 
			size_down=False)
		first_block_params['conv1']['input_channels'] = 1024
		self.conv4_x_2 = ResBlock(
			params=self.conv4_x_params, 
			size_down=False)
		self.conv4_x_3 = ResBlock(
			params=self.conv4_x_params, 
			size_down=False)
		self.conv4_x_4 = ResBlock(
			params=self.conv4_x_params, 
			size_down=False)
		self.conv4_x_5 = ResBlock(
			params=self.conv4_x_params, 
			size_down=False)
		self.conv4_x_6 = ResBlock(
			params=self.conv4_x_params, 
			size_down=True)

		# construct the conv5_x.
		self.conv5_x_params = self.res50_params['conv5_x']
		first_block_params = res50_params['conv5_x']
		first_block_params['conv1']['input_channels'] = 1024
		self.conv5_x_1 = ResBlock(
			params=first_block_params,
			size_down=False)
		first_block_params['conv1']['input_channels'] = 2048
		self.conv5_x_2 = ResBlock(
			params=self.conv5_x_params, 
			size_down=False)
		self.conv5_x_3 = ResBlock(
			params=self.conv5_x_params, 
			size_down=True)

		# construct the last stage.
		self.global_ave_pooling = nn.AvgPool2d(kernel_size=2)
		self.flatten = nn.Flatten()
		self.fc = nn.Linear(
			in_features=2048, 
			out_features=10)

		print(INFO, 'done!')

	def forward(self, x):
		#print(x.shape)
		x = self.conv1_x(x)
		#print(x.shape)
		x = self.norm1(x)
		#print(x.shape)
		x = F.relu(x)
		#print(x.shape)
		x = self.conv2_x_1(x)
		#print(x.shape)
		x = self.conv2_x_2(x)
		#print(x.shape)
		x = self.conv2_x_3(x)
		x = self.conv3_x_1(x)
		x = self.conv3_x_2(x)
		x = self.conv3_x_3(x)
		x = self.conv3_x_4(x)
		x = self.conv4_x_1(x)
		x = self.conv4_x_2(x)
		x = self.conv4_x_3(x)
		x = self.conv4_x_4(x)
		x = self.conv4_x_5(x)
		x = self.conv4_x_6(x)
		x = self.conv5_x_1(x)
		x = self.conv5_x_2(x)
		x = self.conv5_x_3(x)
		x = self.global_ave_pooling(x)
		x = self.flatten(x)
		x = self.fc(x)
		x = F.relu(x)
		#x = F.softmax(x)
		return x

def fit(model, training_params, evaluate_params):
	device = training_params['device']
	epochs = training_params['epochs']
	optimizer = training_params['optimizer']
	loss_fn = training_params['loss_fn']
	now_time = time.asctime(time.localtime(time.time()))
	print(INFO, 'training start at', now_time)
	total_start_time = time.time()
	for epoch in range(epochs):
		model.train()
		if 32000 > epoch > 500:
			optimizer = torch.optim.SGD(
				model.parameters(), 
				lr=0.1, 
				momentum=0.9, 
				weight_decay=0.0001)
		if 48000 > epoch >= 32000:
			optimizer = torch.optim.SGD(
				model.parameters(), 
				lr=0.01, 
				momentum=0.9, 
				weight_decay=0.0001)
		if epoch >= 48000:
			optimizer = torch.optim.SGD(
				model.parameters(), 
				lr=0.001, 
				momentum=0.9, 
				weight_decay=0.0001)
		start_time = time.time()
		training_loss = 0
		data_loader = training_params['data']
		for i, data in enumerate(data_loader, 0):
			training_samples_num = 0
			training_correct_mum = 0
			images, labels = data[0].to(device), data[1].to(device)
			outputs = model(images)
			_, training_predicted = torch.max(outputs, 1)
			training_samples_num = labels.size(0)
			training_correct_num = (training_predicted == labels).sum().item()
			loss = loss_fn(outputs, labels)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			training_loss += loss.item()
			if i % 20 == 19:
				end_time = time.time()
				print('\repoch: %d\ttraining loss: %.5f\t \
					training accuracy: %.3f\tcost time: %.3f'
					% (epoch +1, 
						training_loss/(i+1), 
						(100*training_correct_num/training_samples_num),
						end_time-start_time),
					end='')
				writer.add_scalar('training loss',
					training_loss/(i+1),
					epoch*len(data_loader)+i)
		end_time = time.time()
		print('\repoch: %d\ttraining loss: %.5f\ttraining accuracy: %.3f\
			cost time: %.3f' 
			% (epoch+1, 
				training_loss/(i+1), 
				100*training_correct_num/training_samples_num, 
				end_time-start_time), 
			end='')
		training_loss = 0
		model.eval()
		device = evaluate_params['device']
		data_loader = evaluate_params['data']
		evaluating_samples_num = 0
		evaluating_correct_num = 0
		with torch.no_grad():
			for i, data in enumerate(data_loader, 0):
				images, labels = data[0].to(device), data[1].to(device)
				outputs = model(images)
				_, evaluating_predicted = torch.max(outputs, 1)
				evaluating_samples_num += labels.size(0)
				evaluating_correct_num += (evaluating_predicted == labels).sum().item()
			print('\taccuracy: %.3f' % (100*evaluating_correct_num/evaluating_samples_num), '%')
	total_end_time = time.time()
	now_time = time.asctime(time.localtime(time.time()))
	print('training finished at', now_time)
	print('total time used: %.3f' % (total_end_time-total_start_time))
	torch.save(model.state_dict(), 'mnist_v100.pt')
	print(INFO, 'model saved successfully!')


def main():

	training_data_loader, testing_data_loader, classes = get_data()

	res50_params = set_res50_params()

	res50 = Res50(res50_params)

	print(INFO, 'preparing to load model from res50_v101.pt...')
	try:
		res50.load_state_dict(torch.load('res50_v101.pt'))
		print(INFO, 'load model successfully!')
	except:
		print(INFO, 'failed to load model, maybe there is \
			no file named res50_v101.pt')

	training_params, evaluating_params = set_params(
		res50,
		training_data_loader, 
		testing_data_loader)

	device = torch.device('cuda:3')
	res50.to(device)
	fit(res50, training_params, evaluating_params)
	images, labels = next(iter(training_data_loader))
	writer.add_graph(res50, images.to(device))
	writer.close()


#if '__name__' == __main__:
main()
			





















