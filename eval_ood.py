import os
import os.path
import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from utils.dataloader.folder import PlainDatasetFolder

from torch.autograd import Variable
from torch.utils import data
from PIL import Image
from utils.dataloader.pascal_voc_loader import *
from utils.dataloader.coco_loader import *
import random
# import tqdm
import torchvision.transforms as transforms
#---- your own transformations
from utils.transform import ReLabel, ToLabel, ToSP, Scale

from model.classifiersimple import *
from sklearn import metrics
from skimage.filters import gaussian as gblur
from utils import anom_utils


def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

def create_association_matrix(scores, num_labels, thresh=0.5, normalize=True):
    pos_association_matrix = [[0 for _ in range(num_labels)] for _ in range(num_labels)]
    neg_association_matrix = [[0 for _ in range(num_labels)] for _ in range(num_labels)]

    labels = scores > thresh
    for index, label in enumerate(labels):
        for index2, l in enumerate(label):
            
            if l:
                pos_association_matrix[index2] += scores[index]
            else:
                neg_association_matrix[index2] += scores[index]
    pos_association_matrix = np.array(pos_association_matrix)
    neg_association_matrix = np.array(neg_association_matrix)
    if normalize:
        pos_association_matrix = pos_association_matrix / pos_association_matrix.sum(axis=-1)
        neg_association_matrix = neg_association_matrix / neg_association_matrix.sum(axis=-1)
    return pos_association_matrix, neg_association_matrix 

def get_ood_kl_score_helper(scores, pos_mat, neg_mat, thresh=0.5, normalize=True):
    res = []
    labels = np.array(scores) > thresh
    for index, label in enumerate(labels):
        tmp = 0
        count = 0
        for index2, l in enumerate(label): 
            if l:
                input  = scores[index]
                target = pos_mat[index2]
                #input  = pos_mat[index2]
                #target = scores[index]
                tmp += torch.nn.functional.mse_loss(torch.from_numpy(input), torch.from_numpy(target))
                #tmp -= torch.nn.functional.cosine_similarity(
                #    torch.from_numpy(input), torch.from_numpy(target), dim=-1)
            else:
                input  = scores[index]
                target = neg_mat[index2]
                #input  = neg_mat[index2]
                #target = scores[index]
                #tmp -= torch.nn.functional.mse_loss(torch.from_numpy(input), torch.from_numpy(target))
                #tmp += torch.nn.functional.cosine_similarity(
                #    torch.from_numpy(input), torch.from_numpy(target), dim=-1)
            if normalize:
                count += 1
            else:
                count = 1
        res.append(tmp/count)
    return res


def get_ood_kl_score(scores, out_scores, pos_mat, neg_mat, thresh=0.5):
    res = []
    res.extend(get_ood_kl_score_helper(scores, pos_mat, neg_mat, thresh=thresh))
    res.extend(get_ood_kl_score_helper(out_scores, pos_mat, neg_mat, thresh=thresh)) 
    
    return np.array(res)

    

def get_predictions(loader, model, clsfier, ood="msp", name=None):
    #gts = {i:[] for i in range(0,num_labels)}
    #preds = {i:[] for i in range(0,num_labels)}
    # gts, preds = [], []
    #print("name = ", name, os.path.exists("./logits/"+name+".npy"))
    if (ood != "dropout") and (name is not None) and (os.path.exists("./logits/"+name+".npy")):
        logits = np.load("./logits/"+name+".npy")
        
        if ood == "msp":
            outputs = F.sigmoid(torch.from_numpy(logits).cuda())
            pred = outputs.squeeze().data.cpu().numpy()
            scores = pred.max(axis=1)
        elif ood == "max_logit":
            scores = logits.max(axis=1)
        elif ood == "sum_logit":
            scores = logits.sum(axis=1)
        elif ood == "logit_avg":
            scores = logits.mean(axis=1)
        elif ood == "lof":
            scores = logits
        elif ood == "isol":
            scores = logits
        elif ood == "kl":
            #outputs = F.sigmoid(torch.from_numpy(logits).cuda())
            #scores = outputs.squeeze().data.cpu().numpy()
            scores = logits
        else:
            raise NameError('ood measure not implemented')

    else:
        repeat_amt = 1
        if ood == "dropout":
            model.train()
            clsfier.train()
            repeat_amt = 10


        logits = []
        scores = []

        softmax = nn.Softmax2d()
        for i, (images, labels) in tqdm(enumerate(loader)):
            img_stacks = []
            if torch.cuda.is_available():
                #print(len(images))
                images = Variable(images[0].cuda(), volatile=True)
                #labels = Variable(labels[0].cuda().float())   
            else:
                images = Variable(images[0], volatile=True)
                #labels = Variable(labels[0]) 

            # outputs  = softmax(segcls(model(images)))
            #print(images.size())
            failed = False
            for i in range(repeat_amt):
                outputs = model(images)
                try:
                    outputs = clsfier(outputs)
                except ValueError:
                    failed = True
                if failed:
                    break
                img_stacks.append(outputs.clone())
                outputs_np = outputs.squeeze().data.cpu().numpy()
                outputs = F.sigmoid(outputs)
                pred = outputs.squeeze().data.cpu().numpy()
                
                torch.cuda.empty_cache()
            if failed:
                continue

            logits.append(outputs_np)
            if ood == "msp":
                scores.append(pred.max())
            elif ood == "max_logit":
                scores.append(outputs_np.max())
            elif ood == "max_logit":
                scores.append(outputs_np.sum())
            elif ood == "logit_avg":
                scores.append(outputs_np.mean())
            elif ood == "lof":
                scores.append(outputs_np)
            elif ood == "isol":
                scores.append(outputs_np)
            elif ood == "kl":
                 scores.append(pred)
            elif ood == "dropout":
                img_stacks = torch.stack(img_stacks, dim=0)
                scores.append(img_stacks.var(dim=1).mean(dim=0).squeeze().data.cpu().numpy())
                del img_stacks, outputs
                torch.cuda.empty_cache()
            else:
                raise NameError('ood measure not implemented')
            #gt = labels.squeeze().data.cpu().numpy()
            
            #for label in range(0,num_labels):
            #    gts[label].append(gt[label])
            #    preds[label].append(pred[label])
        if ood != "dropout":
            directory = "./logits/"
            if not os.path.exists(directory):
                os.makedirs(directory)
            np.save(directory+name, logits)

    return scores

def validate(args):
    # Setup Dataloader
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    img_transform = transforms.Compose([
            transforms.Scale((256,256)),
            transforms.ToTensor(),
            normalize,
        ])

    label_transform = transforms.Compose([
            ToLabel(),
            # normalize,
        ])

    if args.dataset == 'pascal':
        loader = pascalVOCLoader('./datasets/pascal/', split=args.split, img_transform = img_transform , label_transform = label_transform)
        test_loader = pascalVOCLoader('./datasets/pascal/', split="voc12-test", img_transform = img_transform, label_transform = None)
    elif args.dataset == 'coco':
        loader = cocoloader('./datasets/coco/', split=args.split,img_transform = img_transform, label_transform = label_transform)
        test_loader = cocoloader('./datasets/coco/', split="test", img_transform = img_transform, label_transform = None)
    else:
        raise AssertionError

    n_classes = loader.n_classes
    valloader = data.DataLoader(loader, batch_size=args.batch_size, num_workers=4, shuffle=False)
    testloader = data.DataLoader(test_loader, batch_size=args.batch_size, num_workers=4, shuffle=False)

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

    save_name = "test" + args.dataset
    in_scores = get_predictions(testloader, model, clsfier, args.ood, name=save_name)

    ood_root = "./datasets/ImageNet-22K/"
    ood_subfolders = ["n02069412", "n02431122", "n02392434", "n02508213", "n01970164", "n01937909", "n12641413", "n12649317", "n12285512", "n11978713",
                      "n07691650", "n07814390", "n12176953", "n12126084", "n12132956", "n12147226", "n12356395", "n12782915", "n02139199", "n01959492"]


    #aurocs = [];
    out_scores = []
    for folder in ood_subfolders:
        root = os.path.join(ood_root, folder)
        imgloader = PlainDatasetFolder(root, transform=img_transform)
        loader = data.DataLoader(imgloader, batch_size=args.batch_size, num_workers=4, shuffle=False)
        save_name = args.dataset+folder
        out_scores.extend(get_predictions(loader, model, clsfier, args.ood, name=save_name))


    if args.ood == "lof":
        save_name = "val" + args.dataset
        val_scores = get_predictions(valloader, model, clsfier, args.ood, name=save_name)

        scores = anom_utils.get_localoutlierfactor_scores(val_scores, in_scores, out_scores)
        in_scores = scores[:len(in_scores)]
        out_scores = scores[-len(out_scores):]

    elif args.ood == "isol":
        save_name = "val" + args.dataset
        val_scores = get_predictions(valloader, model, clsfier, args.ood, name=save_name)

        scores = anom_utils.get_isolationforest_scores(val_scores, in_scores, out_scores)
        in_scores = scores[:len(in_scores)]
        out_scores = scores[-len(out_scores):]

    elif args.ood == "kl":
        save_name = "val" + args.dataset
        #thresholds = [0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9]
        in_scores = np.array(in_scores)
        out_scores = np.array(out_scores)
        thresholds = np.linspace(
            np.max([in_scores.min(), out_scores.min()]) + 0.1,
            np.min([in_scores.max(), out_scores.max()]), num=20, endpoint=False)[2:]
        for thresh in thresholds:
            print(f"thresh = {thresh}")
            val_scores = get_predictions(valloader, model, clsfier, args.ood, name=save_name)
            pos_mat, neg_mat = create_association_matrix(val_scores, n_classes, thresh=thresh)
            scores = get_ood_kl_score(in_scores, out_scores, pos_mat, neg_mat, thresh=thresh)
            tmp_in_scores = scores[:len(in_scores)]
            tmp_out_scores = scores[-len(out_scores):]
            try: 
                auroc, aupr, fpr = anom_utils.get_and_print_results(tmp_in_scores, tmp_out_scores)
            except:
                break
            print("mean auroc = ", np.mean(auroc), "mean aupr = ", np.mean(aupr), " mean fpr = ", np.mean(fpr))



    #if args.flippin:
    #in_scores = - np.asarray(out_scores)
    #out_scores = - np.asarray(in_scores)

    auroc, aupr, fpr = anom_utils.get_and_print_results(in_scores, out_scores)
    #np.save('./in_scores', in_scores)
    #np.save('./out_scores', out_scores)
    #aurocs.append(auroc); auprs.append(aupr), fprs.append(fpr)
    #print(np.mean(aurocs), np.mean(auprs), np.mean(fprs))
    print("mean auroc = ", np.mean(auroc), "mean aupr = ", np.mean(aupr), " mean fpr = ", np.mean(fpr))



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
    parser.add_argument('--ood', type=str, default='msp',
                        help='which measure to use [msp, logit_avg]')
    parser.add_argument('--flippin', action='store_true')
    args = parser.parse_args()
    validate(args)
