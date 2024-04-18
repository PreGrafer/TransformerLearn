import torch
import torch.nn as nn
from torch import optim

from ViT.Utils import get_data_loaders
from ViT.model import ViT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50

BATH_SIZE = 4
TRAIN_DIR = './Data/mnist_train.csv'
TEST_DIR = './Data/mnist_test.csv'

IN_CHANNELS = 1
IMG_SIZE = 28
PATCH_SIZE = 4
EMBED_SIZE = (PATCH_SIZE ** 2) * IN_CHANNELS
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2
DROP_OUT = 0.001

NUM_HEADS = 8
ACTIVATION = nn.GELU()
NUM_ENCODERS = 768
NUM_CLASSES = 10
LEARNING_RATE = 0.0001
ADAM_WEIGHT_DECAY = 0
ADAM_BETAS = (0.9, 0.999)

train_loader, val_loader, test_loader = get_data_loaders(TRAIN_DIR, TEST_DIR, BATH_SIZE)

model = ViT(IN_CHANNELS, PATCH_SIZE, EMBED_SIZE, NUM_PATCHES, DROP_OUT,
          NUM_HEADS, ACTIVATION, NUM_ENCODERS, NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=ADAM_WEIGHT_DECAY, betas=ADAM_BETAS)

for epoch in range(EPOCHS):
    model.train()
    train_labels = []
    train_preds = []
    train_running_loss = 0
    for idx,image_label in enumerate(train_loader):
        image = image_label['image'].float().to(device)
        label = image_label['label'].type(torch.uint8).to(device)
        y_pred = model(image)
        y_pred_label = torch.argmax(y_pred, dim=1)

        train_labels.extend(label.cpu().detach())
        train_preds.extend(y_pred_label.cpu().detach())

        loss = criterion(y_pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_running_loss += loss.item()

    train_loss =  train_running_loss / (idx + 1)

    model.eval()
    val_labels = []
    val_preds = []
    val_running_loss = 0
    val_loss = 0
    with torch.no_grad():
        for idx, image_label in enumerate(val_loader):
            image = image_label['image'].float().to(device)
            label = image_label['label'].type(torch.uint8).to(device)
            y_pred = model(image)
            y_pred_label = torch.argmax(y_pred, dim=1)

            val_labels.extend(label.cpu().detach())
            val_preds.extend(y_pred_label.cpu().detach())

            loss = criterion(y_pred, label)
            val_running_loss += loss.item()

    val_loss = val_running_loss / (idx + 1)

    print('-' * 30)
    print(f"Epoch: {epoch + 1}/{EPOCHS}, "
          f"Train Loss: {train_loss:.4f}, "
          f"Val Loss: {val_loss:.4f}, "
          f"Train_Accuracy: {sum(1 for x,y in zip(train_preds, train_labels) if x == y) / len(train_labels):.4f}, "
          f"Val_Accuracy: {sum(1 for x,y in zip(val_preds, val_labels) if x == y) / len(val_labels):.4f}")
    print('-' * 30)