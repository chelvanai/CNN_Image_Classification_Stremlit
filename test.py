import json

import torch
from PIL import Image

from data_loader import train_transforms
from model import Net

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def test(image, model, labels):
    model.load_state_dict(torch.load('weight/weight.pth'))
    model.eval()

    img = Image.open(image)
    img = img.resize((32, 32), resample=0, box=None)
    image_tensor = train_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = image_tensor.to(device)
    # print(input.shape)
    output = model(input)
    index = output.data.cpu().numpy()
    # print(index)
    image_labels = []
    for key in labels.keys():
        image_labels.append(key)

    res = dict(zip(image_labels, index.tolist()[0]))
    print(res)
    print("Prediction is ", image_labels[index.argmax()])
    return image_labels[index.argmax()]


if __name__ == '__main__':
    model = Net().to(device)

    with open('data.json', 'r') as f:
        labels = json.load(f)

    test("test.jpeg", model, labels)
