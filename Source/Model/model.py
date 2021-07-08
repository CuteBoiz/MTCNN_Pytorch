import torch
from torch import nn

def weights_init(m):
	if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
		nn.init.xavier_uniform(m.weight.data)
		nn.init.constant(m.bias, 0.1)

class PNet(nn.Module):
	def __init__(self, device):
		super().__init__()
		self.device = device
		self.pre_layer = nn.Sequential(
			nn.Conv2d(3, 10, kernel_size=3),
			nn.PReLU(10),
			nn.MaxPool2d(2, 2, ceil_mode=True),
			nn.Conv2d(10, 16, kernel_size=3),
			nn.PReLU(16),
			nn.Conv2d(16, 32, kernel_size=3),
			nn.PReLU(32)
		)
		self.conv4_1 = nn.Conv2d(32, 2, kernel_size=1)
		self.softmax4_1 = nn.Softmax(dim=1)
		self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1)
		self.apply(weights_init)

	def forward(self, x):
		x = self.pre_layer(x).to(self.device, dtype=torch.double)
		face_classify = self.conv4_1(x)
		face_classify = self.softmax4_1(face_classify)
		bb_regress = self.conv4_2(x)
		return bb_regress, face_classify

class RNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.pre_layer = nn.Sequential(
			nn.Conv2d(3, 28, kernel_size=3),
			nn.PReLU(28),
			nn.MaxPool2d(3, 2, ceil_mode=True),
			nn.Conv2d(28, 48, kernel_size=3),
			nn.PReLU(48),
			nn.MaxPool2d(3, 2, ceil_mode=True),
			nn.Conv2d(48, 64, kernel_size=2),
			nn.PReLU(64),
			nn.Linear(576, 128),
			nn.PReLU(128)
		)
		self.dense5_1 = nn.Linear(128, 2)
		self.softmax5_1 = nn.Softmax(dim = 1)
		self.dense5_2 = nn.Linear(128, 4)
		self.appy(weights_init)

	def forward(self):
		x = self.pre_layer(x)
		x = x.permute(0, 3, 2, 1).contiguous()
		x = self.dense4(x.view(x.shape[0], -1))
		x = self.prelu4(x)
		face_classify = self.dense5_1(x)
		face_classify = self.softmax5_1(face_classify)
		bb_regress = self.dense5_2(x)
		return bb_regress, face_classify

class ONet(nn.Module):
	def __init__(self):
		super().__init__()
		self.pre_layer = nn.Sequential(
			nn.Conv2d(3, 32, kernel_size=3),
			nn.PReLU(32),
			nn.MaxPool2d(3, 2, ceil_mode=True),
			nn.Conv2d(32, 64, kernel_size=3),
			nn.PReLU(64),
			nn.MaxPool2d(3, 2, ceil_mode=True),
			nn.Conv2d(64, 64, kernel_size=3),
			nn.PReLU(64),
			nn.MaxPool2d(2, 2, ceil_mode=True),
			nn.Conv2d(64, 128, kernel_size=2),
			nn.PReLU(128)
		)
		self.dense5 = nn.Linear(1152, 256),
		self.prelu5 = nn.PReLU(256)
		self.dense6_1 = nn.Linear(256, 2)
		self.softmax6_1 = nn.Softmax(dim=1)
		self.dense6_2 = nn.Linear(256, 4)
		self.dense6_3 = nn.Linear(256, 10)
		self.apply(weights_init)

	def forward(self):
		x = self.pre_layer(x)
		x = x.permute(0, 3, 2, 1).contiguous()
		x = self.dense5(x.view(x.shape[0], -1))
		x = self.prelu5(x)
		face_classify = self.dense6_1(x)
		face_classify = self.softmax6_1(face_classify)
		bb_regress = self.dense6_2(x)
		landmarks = self.dense6_3(x)
		return bb_regress, face_classify, landmarks






