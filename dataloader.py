import torchvision
from torch.utils.data import DataLoader


def data_loader(train_data_path, test_data_path, train_transform, test_transform, args):
    batch_size = args.batch_size
    train_data = torchvision.datasets.ImageFolder(train_data_path,
                                                  transform=train_transform)
    # train_data_2 = random.shuffle(train_data)
    print("train data size:", len(train_data))
    args.train_size = len(train_data)
    test_data = torchvision.datasets.ImageFolder(test_data_path,
                                                 transform=test_transform)
    print("test data size:", len(test_data))
    test_size = len(test_data) * 1.0
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    # train_loader_2 = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=40, shuffle=True)
    return train_loader, test_loader, test_size
