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
from sklearn import metrics

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


def validate(args):
    # Setup Dataloader
    if args.arch == 'nasnetalarge':
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        img_transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation((-30,30)),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
            transforms.Scale((256,256)),
            # transforms.RandomResizedCrop((256)),
            # transforms.RandomResizedCrop((382),scale=(0.5, 2.0)),
            transforms.ToTensor(),
            normalize,
        ])

    label_transform = transforms.Compose([
            ToLabel(),
            # normalize,
        ])
    if args.dataset == 'pascal':
        num_labels = 20
        loader = pascalVOCLoader('/share/data/vision-greg/mlfeatsdata/CV_Course/', split=args.split, img_transform = img_transform , label_transform = label_transform)
    elif args.dataset == 'coco':
        num_labels = 80
        loader = cocoloader('/share/data/vision-greg/mlfeatsdata/Pytorch/sharedprojects/NIPS-2019/data-convertor/', split=args.split,img_transform = img_transform , label_transform = label_transform)
    else:
        raise AssertionError
    n_classes = loader.n_classes
    valloader = data.DataLoader(loader, batch_size=args.batch_size, num_workers=4, shuffle=True)

    print(len(loader))
    print(normalize)
    print(args.arch)

    orig_resnet = torchvision.models.resnet101(pretrained=True)
    features = list(orig_resnet.children())
    model= nn.Sequential(*features[0:8])
    clsfier = clssimp(2048,n_classes)

    model.load_state_dict(torch.load('savedmodels/' + args.arch + str(args.disc) +  ".pth"))
    clsfier.load_state_dict(torch.load('savedmodels/' + args.arch +'clssegsimp' + str(args.disc) +  ".pth"))

    model.eval()
    clsfier.eval()
    print(len(loader))

    if torch.cuda.is_available():
        model.cuda(0)
        clsfier.cuda(0)

    model.eval()
    gts = {i:[] for i in range(0,num_labels)}
    preds = {i:[] for i in range(0,num_labels)}
    # gts, preds = [], []
    softmax = nn.Softmax2d()
    for i, (images, labels) in tqdm(enumerate(valloader)):
        if torch.cuda.is_available():
            images = Variable(images[0].cuda(0))
            labels = Variable(labels[0].cuda(0).float())
   
        else:
            images = Variable(images[0])
            labels = Variable(labels[0]) 

        # outputs  = softmax(segcls(model(images)))
        
        outputs  = model(images)
        outputs = clsfier(outputs)
        outputs = F.sigmoid(outputs)
        pred = outputs.squeeze().data.cpu().numpy()
        gt = labels.squeeze().data.cpu().numpy()
        #print(gt.shape)
        
        for label in range(0,num_labels):
            gts[label].append(gt[label])
            preds[label].append(pred[label])
        # for gt_, pred_ in zip(gt, pred):
        #     gts.append(gt_)
        #     preds.append(pred_)

    FinalMAPs = []
    
    for i in range(0,num_labels):
        precision, recall, thresholds = metrics.precision_recall_curve(gts[i], preds[i]);
        FinalMAPs.append(metrics.auc(recall , precision));
    print(FinalMAPs)
    print(np.mean(FinalMAPs))



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
    parser.add_argument('--batch_size', nargs='?', type=int, default=1, 
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-4, 
                        help='Learning Rate')
    parser.add_argument('--split', nargs='?', type=str, default="voc12-val", 
                        help='Split of dataset to test on')
    parser.add_argument('--load', nargs='?', type=int)
    parser.add_argument('--alpha', nargs='?', type=float)
    parser.add_argument('--disc', nargs='?', type=str)
    parser.add_argument('--FG', nargs='?', type=int, default=0)
    parser.add_argument('--DT', nargs='?', type=int, default=0)
    parser.add_argument('--BN', nargs='?', type=int, default=1)
    parser.add_argument('--feature_scale', nargs='?', type=int, default=1, 
                        help='Divider for # of features to use')    
    args = parser.parse_args()
    validate(args)
