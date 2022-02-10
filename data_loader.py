import json
import os

import torch
from PIL import Image
from sklearn import preprocessing
from torch.utils.data import Dataset
from torchvision import transforms

means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(means, stds)]
)


class CustomImageDataset(Dataset):
    def __init__(self, img_path):
        self.img_path = img_path

        self.data = []
        self.label = []
        for i in os.listdir(self.img_path):
            if os.path.isdir(self.img_path + "/" + i):
                for j in os.listdir(self.img_path + "/" + i):
                    self.data.append(self.img_path + "/" + i + "/" + j)
                    self.label.append(str(i))
        label_encoder = preprocessing.LabelEncoder()
        self.label = label_encoder.fit_transform(self.label)

        mapping = dict(zip(label_encoder.classes_, list(map(lambda x: int(x), label_encoder.transform(label_encoder.classes_)))))

        with open('data.json', 'w') as f:
            json.dump(mapping, f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx]).convert('RGB')
        img = train_transforms(img)
        label = self.label[idx]
        label_tensor = torch.as_tensor(label, dtype=torch.long)
        return {'im': img, 'labels': label_tensor}
