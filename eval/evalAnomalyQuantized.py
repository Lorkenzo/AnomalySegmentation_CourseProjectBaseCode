# Copyright (c) OpenMMLab. All rights reserved.
import os
#import cv2
import glob
import time
import torch
import random
from PIL import Image
import numpy as np
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from temperature_scaling import ModelWithTemperature
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from dataset import TestDataset
from sklearn.model_selection import train_test_split
from torch.quantization.observer import MinMaxObserver,HistogramObserver
from torchinfo import summary
import torch.nn.utils.prune as prune

#Models
from models.erfnet import ERFNet
from models.erfnetQ import ERFNetQ,DownsamplerBlock
from models.enet import ENet
from models.bisenet import BiSeNetV2

from matplotlib import pyplot as plt
seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 20
# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def transform_label(label):
        return torch.squeeze(label,0).long()   

transform_image = transforms.Resize((704,1280))

def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
              
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]] = param
                else:
                   continue
            else:
                own_state[name] = param 
        return model

def compute_model_stats(model, input_size):
    summary(model, input_size=input_size, col_names=["input_size", "output_size", "num_params", "mult_adds"], depth=0)
     
def apply_pruning(model,image_size, amount=0.2):
    
    # Convolutional layers are the ones more suitable for pruning
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.ConvTranspose2d):
            prune.l1_unstructured(module, name='weight',amount=amount)  # Prune the weights

    compute_model_stats(model, image_size)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.ConvTranspose2d):
            prune.remove(module, 'weight')
    
    return model
     
def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )  
    parser.add_argument('--loadDir',default="trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_quantized.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--void',action='store_true')

    args = parser.parse_args()
    anomaly_score_list = []
    ood_gts_list = []

    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    # Initialize quantized model
    model = ERFNetQ(NUM_CLASSES)

    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    # Load pretrained model
    
    state_dict = torch.load(weightspath, weights_only=False, map_location=lambda storage, loc: storage)

    # get image size
    dataset_path_list = glob.glob(os.path.expanduser(str(args.input[0])))
    image_size = torch.from_numpy(np.array(Image.open(dataset_path_list[0]).convert('RGB'))).unsqueeze(0).float().permute(0,3,1,2).shape

    print ("Model and weights LOADED successfully")
    # Check stats of the model
    print("\n\t\tComputing initial model stats...")
    compute_model_stats(model, image_size)

    start = time.time()
    # QUANTIZATION of the model

    # Create qcofing for the quantization
    qconfig = torch.quantization.get_default_qconfig('x86')

    qconfig = qconfig._replace(
        activation=HistogramObserver.with_args(dtype=torch.quint8),
        weight=MinMaxObserver.with_args(dtype=torch.qint8)
    )

    # Exclude from the quantization not supported layers
    model.qconfig = qconfig
    
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.ConvTranspose2d):
            layer.qconfig = None
        if name == "encoder.output_conv":
            layer.qconfig = None

    model = torch.quantization.prepare(model, inplace=False)
    model = torch.quantization.convert(model, inplace=False) 

    model = load_my_state_dict(model, state_dict )

    # Check new stats of the model
    print("\n\t\tComputing model stats after quantization...")
    compute_model_stats(model, image_size)

    start = time.time()

    model.eval()

    for i,path in enumerate(glob.glob(os.path.expanduser(str(args.input[0])))):
        images = torch.from_numpy(np.array(Image.open(path).convert('RGB'))).unsqueeze(0).float()
        images = images.permute(0,3,1,2)
    
        with torch.no_grad():
            result = model(images)

        
        #take only background as anomaly
        background_index = 19 # background is the last one
        result_void = result[:,background_index,:,:].unsqueeze(0)

        #Max-logit
        anomaly_result_void = - np.max(result_void.squeeze(0).data.cpu().numpy(), axis=0)  
        anomaly_result_full = - np.max(result.squeeze(0).data.cpu().numpy(), axis=0)  

        pathGT = path.replace("images", "labels_masks")                
        if "RoadObsticle21" in pathGT:
           pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
           pathGT = pathGT.replace("jpg", "png")                
        if "RoadAnomaly" in pathGT:
           pathGT = pathGT.replace("jpg", "png")  

        mask = Image.open(pathGT)
        if "bisenet" in modelpath:
            mask = transform_image(mask)
        ood_gts = np.array(mask)

        if "RoadAnomaly" in pathGT:
            ood_gts = np.where((ood_gts==2), 1, ood_gts)
        if "LostAndFound" in pathGT:
            ood_gts = np.where((ood_gts==0), 255, ood_gts)
            ood_gts = np.where((ood_gts==1), 0, ood_gts)
            ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)

        if "Streethazard" in pathGT:
            ood_gts = np.where((ood_gts==14), 255, ood_gts)
            ood_gts = np.where((ood_gts<20), 0, ood_gts)
            ood_gts = np.where((ood_gts==255), 1, ood_gts)

        if 1 not in np.unique(ood_gts):
            continue              
        else:
            ood_gts_list.append(ood_gts)
            if args.void:
                anomaly_score_list.append(anomaly_result_void)
            else:
                anomaly_score_list.append(anomaly_result_full) 

        del result, anomaly_result_full,anomaly_result_void, ood_gts, mask
        torch.cuda.empty_cache()

    file.write( "\n")

    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)

    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))
    
    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))

    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)

    print(f'AUPRC score: {prc_auc*100.0}')
    print(f'FPR@TPR95: {fpr*100.0}')

    file.write(('    AUPRC score:' + str(prc_auc*100.0) + '   FPR@TPR95:' + str(fpr*100.0) ))
    file.close()

    end = time.time()

    print(f"Time for quantization + evaluation: {end-start} s")

if __name__ == '__main__':
    main()