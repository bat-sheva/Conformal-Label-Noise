import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import torch

class PolypDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, image_root, gt_root, trainsize, opt):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.opt = opt

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])

        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        noise_level = self.opt.noise_level
        if self.opt.noise_type == 'uniform':
            noise_level_matrix = torch.ones_like(gt) * noise_level  # TODO: add noise
            noise_matrix = (torch.rand_like(noise_level_matrix) <= noise_level_matrix).int()
            noisy_gt = noise_matrix * (1-gt) + (1-noise_matrix) * gt
        elif self.opt.noise_type == 'nonuniform':
            noise_level_matrix = torch.ones_like(gt) * noise_level
            noise_matrix = (torch.rand_like(noise_level_matrix) <= noise_level_matrix).int()
            noise_matrix[..., 50:80, 50:80] = noise_matrix[..., 50, 50]
            # noise_matrix[..., 150:240, 180:270] = noise_matrix[..., 150, 180]
            noise_matrix[..., 220:300, 300:320] = noise_matrix[..., 250, 300]
            noisy_gt = noise_matrix * (1-gt) + (1-noise_matrix) * gt
        elif self.opt.noise_type == 'nonuniform2':
            noise_matrix = torch.zeros_like(gt).int()
            # noise_matrix[..., 50:120, 50:120] = (torch.rand(1) <= noise_level).int().item()
            noise_matrix[..., 120:250, 120:250] = (torch.rand(1) <= noise_level).int().item()
            # noise_matrix[..., 210:300, 280:320] = (torch.rand(1) <= noise_level).int().item()
            noisy_gt = noise_matrix * (1-gt) + (1-noise_matrix) * gt
        return image, noisy_gt

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True, opt=None):

    dataset = PolypDataset(image_root, gt_root, trainsize, opt=opt)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
