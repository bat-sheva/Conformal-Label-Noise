import os
import argparse

import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import tqdm
from torch.optim import lr_scheduler

from coco.src.helper_functions import WebVisionDetection
from src.helper_functions.helper_functions import mAP, CocoDetection, CutoutPIL, ModelEma, add_weight_decay
from src.models import create_model
from src.loss_functions.losses import AsymmetricLoss
from randaugment import RandAugment
from torch.cuda.amp import GradScaler, autocast
import platform
is_windows = 'windows' in platform.system().lower()

parser = argparse.ArgumentParser(description='PyTorch MS_COCO Training')
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--model-path', default='./MS_COCO_TRresNet_M_224_81.8.pth', type=str)
parser.add_argument('--model-name', default='tresnet_m')
parser.add_argument('--num-classes', default=10)
parser.add_argument('-j', '--workers', default=1 if is_windows else 4, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--image-size', default=32*2, type=int,
                    metavar='N', help='input image size (default: 448)')
parser.add_argument('--thre', default=0.8, type=float,
                    metavar='N', help='threshold value')
parser.add_argument('-b', '--batch-size', default=4 if is_windows else 16, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--print-freq', '-p', default=64, type=int,
                    metavar='N', help='print frequency (default: 64)')


device = 'cuda'

def main():
    args = parser.parse_args()
    args.do_bottleneck_head = False
    # Setup model
    print('creating model...')

    transform = transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                    # normalize, # no need, toTensor does normalization
                                ])
    train_transform = transforms.Compose([
                                      transforms.Resize((args.image_size, args.image_size)),
                                      # CutoutPIL(cutout_factor=0.5),
                                      RandAugment(),
                                      transforms.ToTensor(),
                                      # normalize,
                                  ])

    train_image_path = './webvision/train_images'
    val_image_path = './webvision/val_images'
    train_labels_path = './webvision/train_labels'
    val_labels_path = './webvision/val_labels'
    train_dataset = WebVisionDetection(train_image_path, train_labels_path, transform=train_transform)
    val_dataset = WebVisionDetection(val_image_path, val_labels_path, transform=transform)

    print("len(val_dataset)): ", len(val_dataset))
    print("len(train_dataset)): ", len(train_dataset))
    model = create_model(args).to(device)
    if args.model_path:  # make sure to load pretrained ImageNet model
        state = torch.load(args.model_path, map_location='cpu')
        filtered_dict = {k: v for k, v in state['model'].items() if
                         (k in model.state_dict() and 'head.fc' not in k)}
        model.load_state_dict(filtered_dict, strict=False)
    print('done\n')

    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    # Actuall Training
    train_multi_label_coco(model, train_loader, val_loader, args.lr, args)


def train_multi_label_coco(model, train_loader, val_loader, lr, args):
    ema = ModelEma(model, 0.9997, device=device)  # 0.9997^641=0.82
    model = model.to(device)
    # set optimizer
    Epochs = 100
    Stop_epoch = 160
    weight_decay = 1e-4
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    parameters = add_weight_decay(model, weight_decay)
    optimizer = torch.optim.Adam(params=parameters, lr=lr, weight_decay=0)  # true wd, filter_bias_and_bn
    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=Epochs,
                                        pct_start=0.2)
    highest_mAP = 0
    trainInfoList = []
    scaler = GradScaler()
    for epoch in (range(Epochs)):
        # if epoch > Stop_epoch:
        #     break
        for i, (inputData, target) in enumerate(train_loader):
            inputData = inputData.to(device)
            target = target.to(device) # (batch,3,num_classes)
            target = target.max(dim=1)[0]
            with autocast():  # mixed precision
                output = model(inputData).float()  # sigmoid will be done in loss !
            loss = criterion(output, target)
            model.zero_grad()

            scaler.scale(loss).backward()
            # loss.backward()

            scaler.step(optimizer)
            scaler.update()
            # optimizer.step()

            scheduler.step()

            ema.update(model)
            # store information
            if i % 100 == 0:
                trainInfoList.append([epoch, i, loss.item()])
                print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
                      .format(epoch, Epochs, str(i).zfill(3), str(steps_per_epoch).zfill(3),
                              scheduler.get_last_lr()[0], \
                              loss.item()))

        try:
            if not os.path.exists("models/webvision"):
                os.mkdir("models/webvision")
            torch.save(model.state_dict(), os.path.join(
                f'models/webvision', 'model-{}-{}.ckpt'.format(epoch + 1, i + 1)))
        except:
            pass

        model.eval()
        mAP_score = validate_multi(val_loader, model, ema)
        model.train()
        if mAP_score > highest_mAP:
            highest_mAP = mAP_score
            try:
                torch.save(model.state_dict(), os.path.join(
                    f'models/webvision', 'model-highest.ckpt'))
            except:
                pass
        print('current_mAP = {:.2f}, highest_mAP = {:.2f}\n'.format(mAP_score, highest_mAP))


def validate_multi(val_loader, model, ema_model):
    print("starting validation")
    Sig = torch.nn.Sigmoid()
    preds_regular = []
    preds_ema = []
    targets = []
    for i, (input, target) in enumerate(val_loader):
        target = target
        target = target.max(dim=1)[0]
        # compute output
        with torch.no_grad():
            with autocast():
                output_regular = Sig(model(input.to(device))).cpu()
                output_ema = Sig(ema_model.module(input.to(device))).cpu()

        # for mAP calculation
        preds_regular.append(output_regular.cpu().detach())
        preds_ema.append(output_ema.cpu().detach())
        targets.append(target.cpu().detach())

    mAP_score_regular = mAP(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy())
    mAP_score_ema = mAP(torch.cat(targets).numpy(), torch.cat(preds_ema).numpy())
    print("mAP score regular {:.2f}, mAP score EMA {:.2f}".format(mAP_score_regular, mAP_score_ema))
    return max(mAP_score_regular, mAP_score_ema)


if __name__ == '__main__':
    main()
