from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
import torch
from torch import Tensor
from PIL import Image, ImageFile
import numpy as np
import random
from io import BytesIO
from sklearn.utils import shuffle


class RandomHorizontalFlip(object):
    def __call__(self, sample:dict[str, Image.Image]) -> dict[str, Image.Image]:
        image, depth = sample['image'], sample['depth']

        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        return {'image': image, 'depth': depth}

class RandomChannelSwap(object):
    def __init__(self, probability:float) -> None:
        from itertools import permutations
        self.probability = probability
        self.indices = list(permutations(range(3), 3))

    def __call__(self, sample:dict[str, Image.Image]) -> dict[str, Image.Image]:
        image, depth = sample['image'], sample['depth']

        if random.random() < self.probability:
            image = np.asarray(image)
            image = Image.fromarray(image[...,list(self.indices[random.randint(0, len(self.indices) -1)])])

        return {'image': image, 'depth': depth}

class Resize(object):
    def __init__(self, width:int, height:int) -> None:
        self.size = (width, height)

    def __call__(self, sample:dict[str, Image.Image]) -> dict[str, Image.Image]:
        image, depth = sample['image'], sample['depth']
        image = image.resize(self.size)
        depth = depth.resize(self.size)

        return {'image': image, 'depth': depth}

def loadZipToMem(zip_file:str) -> dict[str, bytes]:
    from zipfile import ZipFile
    # Load zip file to memory
    print('Loading dataset from zip file...', end='')

    input_zip = ZipFile(zip_file)
    data = {name: input_zip.read(name) for name in input_zip.namelist()}
    return data

def getDataFromZip(data:dict[str, bytes]) -> tuple[dict[str, bytes], list[list[str]], list[list[str]]]:
    train = list(
        (
            row.split(',')
            for row in (data['train.csv']).decode('utf-8').split('\n')
            if len(row) > 0
        )
    )
    test = list(
        (
            row.split(',')
            for row in (data['testing.csv']).decode('utf-8').split('\n')
            if len(row) > 0
        )
    )
    train = shuffle(train, n_samples=1000)
    return data, train, test # type: ignore

class depthDatasetMemory(Dataset):
    def __init__(self, data:dict[str, bytes], nyu2_train:list[list[str]], transform:transforms.Compose) -> None:
        self.data, self.nyu_dataset, self.transform = data, nyu2_train, transform

    def __getitem__(self, idx:int) -> dict[str, ImageFile.ImageFile]:
        sample = self.nyu_dataset[idx]
        image = Image.open(BytesIO(self.data[sample[0]]))
        depth = Image.open(BytesIO(self.data[sample[1].replace('\r','')]))
        sample = {'image': image, 'depth': depth}
        sample = self.transform(sample)
        return sample # type: ignore
    
    def __len__(self) -> int:
        return len(self.nyu_dataset)

class ToTensor(object):
    def __call__(self, sample:dict[str, Image.Image|np.ndarray]) -> dict[str, Tensor]:
        image, depth = sample['image'], sample['depth']

        image = self.to_tensor(image)
        depth = self.to_tensor(depth).float() * 1000
        depth = torch.clamp(depth, 10, 1000)

        return {'image': image, 'depth': depth}

    def to_tensor(self, pic:Image.Image|np.ndarray) -> Tensor:
        # Case for numpy array
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img.float().div(255)

        # Case for PIL image
        if pic.mode == 'I':
            img = torch.from_numpy(np.asarray(pic, np.int32))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.asarray(pic, np.uint16).astype(np.float32) / 65535)
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img

def getNoTransform() -> transforms.Compose:
    return transforms.Compose([
        ToTensor()
    ])

def getDefaultTrainTransform() -> transforms.Compose:
    return transforms.Compose([
        RandomHorizontalFlip(),
        RandomChannelSwap(0.5),
        ToTensor()
    ])

def getTrainingTestingData(data: dict[str, bytes], batch_size: int) -> tuple[DataLoader, DataLoader]:
    data, train, test = getDataFromZip(data)
    transformed_training = depthDatasetMemory(
        data, train, transform=getDefaultTrainTransform()
    )
    transformed_test = depthDatasetMemory(
        data, test, transform=getNoTransform()
    )
    return DataLoader(transformed_training, batch_size, shuffle=True), DataLoader(transformed_test, batch_size, shuffle=False)
