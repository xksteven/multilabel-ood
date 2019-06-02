import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data
from PIL import Image
from utils.dataloader.pascal_voc_loader import *
from utils.dataloader.coco_loader import *
from scipy.misc import toimage
import random
# import tqdm
import torchvision.transforms as transforms
#---- your own transformations
from utils.transform import ReLabel, ToLabel, ToSP, Scale

from model.classifiersimple import *
import torchvision

def train(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    img_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation((-30,30)),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
            # transforms.Scale((256,256)),
            # transforms.RandomResizedCrop((256)),
            transforms.RandomResizedCrop((256),scale=(0.5, 2.0)),
            transforms.ToTensor(),
            normalize,
        ])

    label_transform = transforms.Compose([
            ToLabel(),
        ])

    if args.dataset == "pascal":
        loader = pascalVOCLoader(
                                 "./datasets/pascal/", 
                                 img_transform = img_transform, 
                                 label_transform = label_transform)
    elif args.dataset == "coco":
        loader = cocoloader(
                            "./datasets/coco/", 
                             img_transform = img_transform, 
                             label_transform = label_transform)
    else:
        raise AssertionError
    n_classes = loader.n_classes
    trainloader = data.DataLoader(loader, batch_size=args.batch_size, num_workers=4, shuffle=True)


    print("number of images = ", len(loader))
    print("number of classes = ", n_classes, " architecture used = ", args.arch)


    orig_resnet = torchvision.models.resnet101(pretrained=True)
    features = list(orig_resnet.children())
    model= nn.Sequential(*features[0:8])
    clsfier = clssimp(2048,n_classes)


    if args.load == 1:
        model.load_state_dict(torch.load('savedmodels/' + args.arch + str(args.disc) +  ".pth"))
        clsfier.load_state_dict(torch.load('savedmodels/' + args.arch +'clssegsimp' + str(args.disc) +  ".pth"))

    if torch.cuda.is_available():
        model.cuda(0)
        clsfier.cuda(0)


    freeze_bn_affine = 1
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            if freeze_bn_affine:
                m.weight.requires_grad = False
                m.bias.requires_grad = False


    optimizer = torch.optim.Adam([{'params': model.parameters(),'lr':args.l_rate/10},{'params': clsfier.parameters()}], lr=args.l_rate)
    # optimizer = torch.optim.Adam([{'params': clsfier.parameters()}], lr=args.l_rate)


    bceloss = nn.BCEWithLogitsLoss()
    for epoch in range(args.n_epoch):
        for i, (images, labels) in enumerate(trainloader):
            if torch.cuda.is_available():
                images = Variable(images[0].cuda(0))
                labels = Variable(labels[0].cuda(0).float())
            else:
                images = Variable(images[0])
                labels = Variable(labels[0])-1

            # iterartion = len(trainloader)*epoch + i
            # poly_lr_scheduler(optimizer, args.l_rate, iteration)
            optimizer.zero_grad()
         
            outputs = model(images)
            outputs = clsfier(outputs)
            loss = bceloss(outputs, labels)  #-- pascal labels

            loss.backward()
            optimizer.step()

            if (i+1) % 20 == 0:
                print("Epoch [%d/%d] Loss: %.4f" % (epoch+1, args.n_epoch, loss.data))


        torch.save(model.state_dict(), args.save_dir + args.arch + str(args.disc) +  ".pth")
        torch.save(clsfier.state_dict(), args.save_dir + args.arch +'clssegsimp' + str(args.disc) +  ".pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='resnet101', 
                        help='Architecture to use [\'fcn32s, unet, segnet etc\']')
    parser.add_argument('--model_path', nargs='?', type=str, default='zoomoutscratch_pascal_1_6.pkl', 
                        help='Path to the saved model')
    parser.add_argument('--dataset', nargs='?', type=str, default='pascal', 
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=352, 
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=352, 
                        help='Height of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=80, 
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=40, 
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-4, 
                        help='Learning Rate')
    parser.add_argument('--load', nargs='?', type=int)
    parser.add_argument('--disc', nargs='?', type=str)
    parser.add_argument("--save_dir", type=str, default="./savedmodels/")
    args = parser.parse_args()
    train(args)
