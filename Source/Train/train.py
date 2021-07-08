import os 
import argparse 
from torch import nn
from torch import optim
from prepare_data import *
from torchvision import transforms
from torch.utils.data import DataLoader

import sys
sys.path.append('../Model/')
from model import PNet, RNet, ONet


anno_filename = "../../Dataset/wider_face_split/wider_face_train_bbx_gt.txt"
data_path = "../../Dataset/WIDER_train/images"


def train(args):
	if torch.cuda.is_available():
		device = torch.device("cuda")
		torch.cuda.set_device(args.cuda)
	else:
		device = torch.device("cpu")

	if args.net == "pnet":
		model = PNet(device)
	elif args.net == "rnet":
		model = RNet()
	elif args.net == "onet": 
		model = ONet()
	else:
		raise Exception("Net Type Error!")

	loss_func = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), args.lr, args.momentum)

	transformed_data = WIDER_Dataset(data_path, anno_filename, transforms.Compose([Resize((12, 12)), Normalize() , To_Tensor()]))
	trainloader = DataLoader(transformed_data, batch_size=1, shuffle=True,
						 collate_fn=transformed_data.collate_fn, num_workers=4,
						 pin_memory=True)

	#model.to(device=device)
	for epoch in range(args.epoch):
		model.train()
		for i_batch, (images, boxes) in enumerate(trainloader):
			images.type(torch.DoubleTensor) 
			images.to(device=device)
			boxes[0].to(device=device, dtype=torch.float)

			output = model(images)
			ptint(output.cpu())


			
			
			




if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--net', choices=['pnet', 'rnet', 'onet'], help="PNet/RNet/Onet train")
	#parser.add_argument('--dataset_path', required=True, help="Path to train dataset folder")
	parser.add_argument('--lr', type=float, default=0.001)
	parser.add_argument('--momentum', type=float, default=0.9)
	parser.add_argument('--epoch', type=int, default=10000)
	parser.add_argument('--cuda', type=int, default=0)
	parser.add_argument('--batch_size', type=int, default=32)
	args = parser.parse_args()
	train(args)


