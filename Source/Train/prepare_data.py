import os
import numpy as np
import cv2
import torch
from skimage import io, transform
from torch.utils.data import Dataset



class WIDER_FACE_Image:
	def __init__(self, path, nrof_faces, boxes):

		assert path is not None
		assert nrof_faces is not None
		assert boxes is not None

		self.path = path
		self.nrof_faces = nrof_faces
		self.boxes = boxes


class WIDER_Dataset(Dataset):
	def __init__(self, data_path, anno_file_path, transform=None):
		assert data_path is not None

		if not os.path.exists(anno_file_path):
			raise Exception("Could not found {}".format(anno_file_path))

		self.transform = transform
		self.WIDER_Face = []

		with open(anno_file_path, 'r') as f:
			lines = f.readlines()
		i = 0
		while i < len(lines):
			path = os.path.join(data_path, (lines[i].rstrip("\n")))
			i += 1
			nrof_faces = int(lines[i])
			i += 1
			pos = []
			for j in range(i, i + np.max((nrof_faces, 1))):
				substring = lines[j].split(" ")
				pos.append(list(map(int, substring[0:4])))
				i += 1
			self.WIDER_Face.append(WIDER_FACE_Image(path, nrof_faces, pos))


	def __len__(self):
		return len(self.WIDER_Face)

	def __getitem__(self, idx: int):
		assert idx < len(self.WIDER_Face)
		image = cv2.imread(self.WIDER_Face[idx].path)
		boxes = self.WIDER_Face[idx].boxes

		sample = {'image': image, 'boxes': boxes}

		if self.transform:
			sample = self.transform(sample)

		return sample

	def collate_fn(self, batch):
		images = list()
		boxes = list()

		for b in batch:
			images.append(b['image'])
			boxes.append(b['boxes'])

		images = torch.stack(images, dim=0)

		return images, boxes


class Resize(object):
	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		self.output_size = output_size
		
	def __call__(self, sample):
		image, boxes = sample['image'], sample['boxes']
		h, w = image.shape[:2]
		if isinstance(self.output_size, int):
			if h > w:
				new_h, new_w = self.output_size * h / w, self.output_size
			else:
				new_h, new_w = self.output_size, self.output_size * w / h
		else:
			new_h, new_w = self.output_size

		resized_img = cv2.resize(src=image, dsize=(int(new_w), int(new_h)), interpolation = cv2.INTER_AREA)
		boxes = np.array(boxes) * [new_w/w, new_h/h, new_w/w, new_h/h]
		return {'image': resized_img, 'boxes': boxes}

class Normalize(object):
	def __call__(self, sample):
		mean = [0.485, 0.456, 0.406]
		std = [0.229, 0.224, 0.225]
		image, boxes = sample['image'], sample['boxes']
		normalized = (image - mean) / std
		return {'image': normalized.astype(float), 'boxes': boxes}

class To_Tensor(object):
	def __call__(self, sample):
		image, boxes = sample['image'], sample['boxes']

		# swap color axis because
		# numpy image: H x W x C
		# torch image: C X H X W
		
		image = image.transpose((2, 0, 1))
		try:
			boxes = torch.from_numpy(boxes).to(torch.float32)
		except TypeError:
			boxes = torch.tensor(boxes).to(torch.float32)
		return {'image': torch.from_numpy(image),
				'boxes': boxes}
					