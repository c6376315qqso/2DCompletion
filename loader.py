import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import random
import cv2
from PIL import Image
import os
import copy
import threading

LENX = 512
LENY = 512
NEED_CREATE = 1
sem = threading.Semaphore(10)


def del_and_noise(dir, index, folder):
    with sem:
        try:
            image = cv2.imread(dir)
            image = cv2.resize(image, (LENX, LENY))
            rand_scale = abs(random.gauss(0.5, 2))
            rand_x = np.random.randint(0, LENX, int(LENX*LENY*rand_scale))
            rand_y = np.random.randint(0, LENY, int(LENY*LENX*rand_scale))
            image[rand_x, rand_y, :] = (0, 0, 0)
            noise = np.random.normal(0, 10, image.shape)
            image = image + noise
            cv2.imwrite(os.path.join(folder, str(index) + '.jpg'), image)
        except:
            print('error in ' + dir)



class dataset(Dataset):

    def __init__(self):
        cnt = 0
        self.folder = '../imagenet/dataset/'
        self.X = []
        self.Y = []
        threads = []
        for rt, dirs, nondirs in os.walk(self.folder):
            for name in nondirs:
                dir = os.path.join(self.folder, name)
                if name[:-4].isdigit():
                    continue
                if dir.endswith('.jpg'):
                    if (NEED_CREATE):
                        t = threading.Thread(
                            target=del_and_noise, args=(dir, cnt, self.folder))
                        t.start()
                        threads += [t]
                    self.X += [os.path.join(self.folder, str(cnt) + '.jpg')]
                    self.Y += [dir]
                    cnt += 1
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0,), std=(1,))
        ])
        for t in threads:
            t.join()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        if (index >= len(self.X)):
            index = 0
        try:
            x = Image.open(self.X[index])
            y = Image.open(self.Y[index])
            y = y.convert('RGB')
            x = x.convert('RGB')
            y = y.resize((LENX, LENY))
        except:
            return self.__getitem__(random.randint(0, 5000))
        return self.transform(x), self.transform(y)


if __name__ == '__main__':
    del_and_noise('timg.jpg', 1, '.')