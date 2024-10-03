import os
from copy import deepcopy
import random
import time
from copy import deepcopy
from typing import Tuple, Any

import numpy as np
import torchvision.datasets
import tqdm
from PIL import Image
from torchvision import datasets as datasets
import torch
from PIL import ImageDraw
from pycocotools.coco import COCO


def parse_args(parser):
    # parsing args
    args = parser.parse_args()
    if args.dataset_type == 'OpenImages':
        args.do_bottleneck_head = True
        if args.th == None:
            args.th = 0.995
    else:
        args.do_bottleneck_head = False
        if args.th == None:
            args.th = 0.7
    return args


def average_precision(output, target):
    epsilon = 1e-8

    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i


def mAP(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        # compute average precision
        ap[k] = average_precision(scores, targets)
    return 100 * ap.mean()


class AverageMeter(object):
    def __init__(self):
        self.val = None
        self.sum = None
        self.cnt = None
        self.avg = None
        self.ema = None
        self.initialized = False

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def initialize(self, val, n):
        self.val = val
        self.sum = val * n
        self.cnt = n
        self.avg = val
        self.ema = val
        self.initialized = True

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
        self.ema = self.ema * 0.99 + self.val * 0.01


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_seed():
    return np.random.randint(0, 2 ** 31)


class CIFARDetection(torchvision.datasets.VisionDataset):
    def __init__(self, images, labels, transform=None):

        seed = get_seed()
        set_seeds(0)
        idx = np.random.permutation(len(images))
        self.images = images[idx]
        self.labels = labels[idx]
        self.transform = transform
        width, height = images[0].shape[:2]

        top1_idxs = np.random.permutation(len(images))
        top2_idxs = np.random.permutation(len(images))
        bot1_idxs = np.random.permutation(len(images))
        bot2_idxs = np.random.permutation(len(images))

        self.collage_images = np.zeros((len(images), 2 * width, 2 * height, 3), dtype=self.images.dtype)
        self.collage_labels = np.zeros((len(images), 3, 10), dtype=self.images.dtype)
        for i in range(len(self.collage_images)):
            top1_idx = top1_idxs[i]
            top2_idx = top2_idxs[i]
            bot1_idx = bot1_idxs[i]
            bot2_idx = bot2_idxs[i]
            new_image = np.zeros((width * 2, height * 2, 3), dtype=images[0].dtype)
            new_image[:width, :height] = self.images[top1_idx]
            new_image[:width, height:] = self.images[top2_idx]
            new_image[width:, height:] = self.images[bot1_idx]
            new_image[width:, :height] = self.images[bot2_idx]
            # import matplotlib.pyplot as plt
            self.collage_labels[i, 0, [self.labels[top1_idx], self.labels[top2_idx],
                                       self.labels[bot1_idx], self.labels[bot2_idx]]] = 1
            self.collage_images[i] = new_image
        set_seeds(seed)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = self.collage_images[index]
        target = self.collage_labels[index]

        if self.transform is not None:
            image = self.transform(Image.fromarray(image).convert('RGB'))

        return image, target

    def __len__(self) -> int:
        return len(self.collage_labels)


class WebVisionDetection(torchvision.datasets.VisionDataset):
    def __init__(self, images_path, labels_path, transform=None):

        seed = get_seed()
        set_seeds(0)
        self.transform = transform
        self.labels = labels_path
        self.images_path = images_path
        top_idx = ((self.labels == 0) | (self.labels == 2) | (self.labels == 3) | (self.labels == 5)).nonzero()[0]
        bot_idx = ((self.labels == 1) | (
                self.labels == 6) | (self.labels == 7) | (self.labels == 8) | (self.labels == 9)).nonzero()[0]
        initial_top_idx_size = len(top_idx)
        self.colage_labels = np.zeros((len(top_idx) // 2, 3, 10))
        i = 0
        self.idx_to_colage_map = {}
        while i < initial_top_idx_size // 2:
            if len(top_idx) < 2 or len(bot_idx) < 2:
                break
            top1_idx = top_idx[0]
            top2_idx = top_idx[1]
            j = 1
            while True:
                if self.labels[top1_idx] != self.labels[top2_idx]:
                    break
                j += 1
                if j >= len(top_idx):
                    top_idx = top_idx[top_idx != top_idx[0]]
                    break
                top2_idx = top_idx[j]
            if self.labels[top1_idx] == self.labels[top2_idx]:
                continue
            top_idx = top_idx[(top_idx != top1_idx) & (top_idx != top2_idx)]
            bot1_idx = bot_idx[0]
            bot2_idx = bot_idx[1]
            j = 1
            while True:
                if self.labels[bot1_idx] != self.labels[bot2_idx]:
                    break
                j += 1
                if j >= len(bot_idx):
                    bot_idx = bot_idx[bot_idx != bot_idx[0]]
                    break
                bot2_idx = bot_idx[j]
            if self.labels[bot1_idx] == self.labels[bot2_idx]:
                continue
            bot_idx = bot_idx[(bot_idx != bot1_idx) & (bot_idx != bot2_idx)]
            self.idx_to_colage_map[i] = [self.labels[top1_idx], self.labels[top2_idx],
                                         self.labels[bot1_idx], self.labels[bot2_idx]]
            # import matplotlib.pyplot as plt
            self.colage_labels[i, 0, [self.labels[top1_idx], self.labels[top2_idx],
                                      self.labels[bot1_idx], self.labels[bot2_idx]]] = 1
            i += 1
        self.colage_labels = torch.Tensor(self.colage_labels[:i])
        set_seeds(seed)

    def load_image(self, index) -> np.ndarray:
        pass

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        top1_idx, top2_idx, bot1_idx, bot2_idx = self.idx_to_colage_map[index]
        top1_image = self.load_image(top1_idx)
        top2_image = self.load_image(top2_idx)
        bot1_image = self.load_image(bot1_idx)
        bot2_image = self.load_image(bot2_idx)
        width, height = top1_image.shape[:2]
        image = np.zeros((width * 2, height * 2, 3), dtype=top1_image.dtype)
        image[:width, :height] = top1_image
        image[:width, height:] = top2_image
        image[width:, height:] = bot1_image
        image[width:, :height] = bot2_image

        target = self.colage_labels[index]

        if self.transform is not None:
            image = self.transform(Image.fromarray(image).convert('RGB'))

        return image, target

    def __len__(self) -> int:
        return len(self.colage_labels)


class CocoDetection(torchvision.datasets.VisionDataset):
    def __init__(self, root, annFile, transform=None, target_transform=None, noise_type=None, noise_level=None,
                 is_clean=None):
        self.root = root
        self.coco = COCO(annFile)

        self.ids = list(self.coco.imgToAnns.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.cat2cat = dict()
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)
        self.noise_level = noise_level
        self.noise_type = noise_type
        self.is_clean = is_clean
        # print(self.cat2cat)
        self.noise_transition_map = {
            2: [4],
            4: [2],
            3: [6, 8],
            6: [3, 8],
            8: [3, 6],
            5: [16],
            16: [5],
            19: [20, 21],
            20: [19, 21],
            21: [19, 20],
            27: [31, 33],
            31: [27, 33],
            33: [27, 31],
            36: [41, 42],
            41: [36, 42],
            42: [41, 36],
            48: [49, 50],
            49: [48, 50],
            50: [49, 48],
            79: [80, 81],
            80: [79, 81],
            81: [80, 79],
        }
        self.translated_noise_transition_map = {}
        for key in self.noise_transition_map:
            translated_key = self.cat2cat[key]
            self.translated_noise_transition_map[translated_key] = list(
                map(lambda x: self.cat2cat[x], self.noise_transition_map[key]))

        old_seed = get_seed()
        set_seeds(0)
        self.transition_matrix = torch.zeros(80, 80).int()
        for i in range(80):
            self.transition_matrix[i] = torch.from_numpy(np.random.permutation(80))
            if i in self.translated_noise_transition_map:
                self.transition_matrix[i, self.translated_noise_transition_map[i]] = 81

        set_seeds(0)
        self.partial_noised_labels = np.random.choice(80, size=(50,), replace=False)

        set_seeds(0)
        if self.noise_type.endswith("even") and not self.noise_type.endswith("uneven"):
            self.noise_level_vec = torch.ones(80) * self.noise_level
        elif self.noise_type.endswith("uneven"):
            weaker_noise_labels = np.random.choice(80, size=(40,), replace=False)
            self.noise_level_vec = torch.ones(80) * self.noise_level
            self.noise_level_vec[weaker_noise_labels] = self.noise_level / 2
        else:
            raise Exception("invalid noise type")

        set_seeds(old_seed)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        output = torch.zeros((3, 80), dtype=torch.long)
        seed = get_seed()
        set_seeds(0)
        for obj in target:
            if obj['area'] < 32 * 32:
                output[0][self.cat2cat[obj['category_id']]] = 1
            elif obj['area'] < 96 * 96:
                output[1][self.cat2cat[obj['category_id']]] = 1
            else:
                output[2][self.cat2cat[obj['category_id']]] = 1

        if not self.is_clean:
            output = output.float()
            if self.noise_type.startswith('independent'):
                noise_mat = (torch.rand(80) < self.noise_level_vec)
                rnd_area_idx = np.random.randint(0, 3, (
                noise_mat.float().sum().int().item(),))  # areas of objects noised as True
                output = output * (1 - noise_mat.float())
                output[rnd_area_idx, noise_mat.bool()] = 1
            # elif self.noise_type == 'dependent':
            #     for obj, noised_objects in self.translated_noise_transition_map.items():
            #         if output[:, obj].sum().item() > 0.5 and output[:, noised_objects].sum().item() < 0.5:
            #             if torch.rand(1).item() < self.noise_level:
            #                 noised_object = np.random.randint(0, len(noised_objects))
            #                 area_idx = output[:, obj].argmax().item()
            #                 output[:, obj] = 0
            #                 output[area_idx, noised_object] = 1
            elif self.noise_type.startswith('partial'):
                noise_probabilities = self.noise_level_vec.clone()
                noise_probabilities[~self.partial_noised_labels] = 0
                noise_mat = (torch.rand(80) < noise_probabilities).float()
                rnd_area_idx = np.random.randint(0, 3,
                                                 (noise_mat.sum().int().item(),))  # areas of objects noised as True
                output = output * (1 - noise_mat)
                output[rnd_area_idx, noise_mat.bool()] = 1
            elif self.noise_type.startswith('dependent'):  # dependent_uneven
                appearing_labels = output.sum(dim=0) > 0.5
                labels_to_flip = (torch.rand(80) < self.noise_level_vec).int()
                labels_to_flip[~appearing_labels] = 0
                appearing_labels_to_flip = labels_to_flip.nonzero().squeeze(-1).tolist()
                if len(appearing_labels_to_flip) > 0:
                    for appearing_label_to_flip in appearing_labels_to_flip:
                        curr_transition_vector = self.transition_matrix[appearing_label_to_flip].clone()
                        curr_transition_vector[appearing_labels] = 0
                        new_label = torch.argmax(curr_transition_vector).item()
                        appearing_labels[new_label] = True
                        labels_to_flip[new_label] = 1
                    rnd_area_idx = np.random.randint(0, 3, (labels_to_flip.sum().int().item(),))
                    output = output * (1 - labels_to_flip.float())
                    output[rnd_area_idx, labels_to_flip.bool()] = 1
            else:
                raise Exception("invalid noise type")
        set_seeds(seed)
        output = output.long()
        target = output

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self) -> int:
        return len(self.ids)


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                    ema_v = ema_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class CutoutPIL(object):
    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return x


def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]
