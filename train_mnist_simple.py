import warnings
import torch
from torch import nn
from torchvision import datasets, transforms, models
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from tqdm import tqdm
import os
from torchvision.models.mobilenetv2 import MobileNetV2
from config import TrainConfig
import time


# UTILITY FUNCTIONS
def load_mobilenet_v2_from_pretrained(classes, model_path):
    if os.path.exists(model_path):
        state_dict = torch.load(model_path)
        _model = MobileNetV2(num_classes=1000)
        _model.load_state_dict(state_dict)
    else:
        warnings.warn(f'Cannot load model as model file does not exist: {model_path}'
                      f'\nTrying to download from web.')
        _model = models.mobilenet_v2(pretrained=True)

    # Adjust the FC layer
    _model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(_model.last_channel, classes),
        )
    return _model


def save_model_state_dict(model, full_file_name):
    try:
        state_dict = model.state_dict()
    except:
        state_dict = model.module.state_dict()

    torch.save(state_dict, full_file_name)


# TRAIN and TEST LOOPS
def train(model, device, train_loader, optimizer, epoch, total_epochs):
    model.train()
    total_loss = 0
    correct = 0

    with tqdm(total=len(train_loader.dataset), unit=' images') as pbar:
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            pbar.set_description(f'Training Epoch: {epoch} / {total_epochs}')
            pbar.set_postfix({'loss': round(loss.item(), 5)})
            pbar.update(train_loader.batch_sampler.batch_size)
            pbar.refresh()


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = round(100. * correct / len(test_loader.dataset), 4)

    print('\nTest set. Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), round(test_acc, 2)))

    return test_acc, test_loss


if __name__ == '__main__':

    config = TrainConfig()

    # Training setup
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    epochs = config.EPOCHS
    sched_step = config.SCHED_STEP
    batch_size = config.BATCH_SIZE
    lr = config.LR
    shuffle_train = config.SHUFFLE_TRAIN
    data_dir = config.DATA_DIR
    model_dir = config.MODEL_DIR

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.repeat(3, 1, 1))])

    trainset = datasets.EMNIST(root=data_dir, train=True, split='mnist', transform=transform, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle_train)

    testset = datasets.EMNIST(root=data_dir, train=False, split='mnist', transform=transform, download=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    model = load_mobilenet_v2_from_pretrained(classes=10,
                                              model_path=model_dir)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=sched_step, gamma=0.05)

    model = model.to(device)

    # Main training loop
    final_acc = 0
    final_loss = 0
    for epoch in range(1, epochs + 1):
        train(model, device, trainloader, optimizer, epoch, epochs)
        final_acc, final_loss = test(model, device, testloader)
        scheduler.step()

    if config.SAVE_FINAL_MODEL:
        if not os.path.exists(config.CHECKPOINT_DIR):
            try:
                os.makedirs(config.CHECKPOINT_DIR)
            except:
                print(f'Model saving directory does not exist and cannot be created: {config.CHECKPOINT_DIR}')
        model_out_dir = os.path.join(config.CHECKPOINT_DIR, f'final_model_TS-{time.time()}.pt')
        save_model_state_dict(model, full_file_name=model_out_dir)
        print(f'Model saved: {model_out_dir}')

    print(f'# Total Epochs   : {epochs}\n'
          f'# Final Accuracy : {final_acc}%\n'
          f'# Final Loss     : {round(final_loss,5)}')
