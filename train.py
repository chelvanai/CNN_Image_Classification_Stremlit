import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_loader import CustomImageDataset
from hparams import learning_rate, num_epochs, batch_size, classes
from model import Net


def train(device, dataloader, model, criterion, optimizer):
    best_train_loss = 10
    for epoch in range(num_epochs):
        print("EPOCH:", epoch, end=" ")
        running_loss = 0
        running_acc = 0

        for i, data in enumerate(dataloader):
            inputs, labels = data['im'].to(device), data['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            result = torch.argmax(outputs, dim=1)
            running_loss += loss.item()
            running_acc += torch.mean((result == labels).type(torch.float))

            loss.backward()
            optimizer.step()

        else:
            train_loss = running_loss / len(dataloader)
            train_acc = running_acc / len(dataloader)

            print("Training Loss: {:.3f}".format(train_loss), end=" ")
            print("Train Accuracy: {:.2f}%".format(train_acc.item() * 100))

            if train_loss < best_train_loss:
                best_train_loss = train_loss
                print("weight saved!")
                torch.save(model.state_dict(), './weight/weight.pth')


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print("CLASSES OF TRAIN DATA", classes)

    dataset = CustomImageDataset("./images")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Net().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    train(device, dataloader, model, criterion, optimizer)
