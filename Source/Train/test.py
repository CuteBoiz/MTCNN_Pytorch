from prepare_data import *
from torchvision import transforms
from torch.utils.data import DataLoader
import cv2

anno_filename = "../../Dataset/wider_face_split/wider_face_train_bbx_gt.txt"
data_path = "../../Dataset/WIDER_train/images"

transformed_data = WIDER_Dataset(data_path, anno_filename, transforms.Compose([Resize((768, 1024)), To_Tensor()]))
trainloader = DataLoader(transformed_data, batch_size=4, shuffle=True,
						 collate_fn=transformed_data.collate_fn, num_workers=4,
						 pin_memory=True)

for i_batch, (images, boxes) in enumerate(trainloader):
	print(images.shape)
	images = images.permute(0, 2, 3, 1)
	images = images.numpy()
	
	for i in range(len(images)):
		image = np.ascontiguousarray(images[i], dtype=np.uint8)
		for box in boxes[i]:
			box = box.numpy()
			x = int(box[0])
			y = int(box[1])
			w = int(box[2])
			h = int(box[3])
			print (x, y, w, h)
			cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 1)
		cv2.imshow('preview', image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		

