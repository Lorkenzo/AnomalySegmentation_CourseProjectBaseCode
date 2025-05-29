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

#Models
from models.erfnet import ERFNet
from models.enet import ENet
from models.bisenet import BiSeNetV2

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

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image 

def plot_anomaly_map(modelpath, image_path, label_path, anomaly_map, anomaly_map_full):
    """
    image_path: percorso dell'immagine originale
    anomaly_map: numpy array 2D con punteggi di anomalia (valori tra 0 e 1)
    """
    # Carica immagine originale
    image = Image.open(image_path).convert('RGB')

    if "bisenet" in modelpath:
            if "RoadAnomaly" in image_path:
                # Needed for bisenet in order to have dimensions multiple of 32 (cause the model downsample the images by 32)
                transform_label = transforms.Resize((704,1280))
                image = transform_label(image)
            elif "RoadObsticle" in image_path:
                transform_label = transforms.Resize((1056,1920))
                image = transform_label(image)

    image = np.array(image)
   
    anomaly_map_resized = anomaly_map

    overlay = np.zeros_like(image, dtype=np.uint8)
    overlay[label_path == 255] = [255, 255, 255]
    overlay[label_path == 1] = [255, 0, 0]

    plt.figure(figsize=(15,5))
    
    # Immagine originale
    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis('off')

    # Anomaly map (grayscale)
    plt.subplot(1, 4, 2)
    plt.imshow(overlay)
    plt.title("Anomaly Label")
    plt.axis('off')

    # Anomaly map (colormap heat)
    plt.subplot(1, 4, 3)
    plt.imshow(anomaly_map_full, cmap="gray")
    plt.title("Anomaly Map")
    plt.axis('off')

    plt.subplot(1, 4, 4)
    
    plt.imshow(anomaly_map_resized, cmap="gray")
    plt.title("Anomaly Map Void")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def transform_label_temp(label):
    return torch.squeeze(label, 0).long()  
     
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
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--temp',type=float, default=None)
    parser.add_argument('--void', action='store_true')

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

    if "erfnet" in modelpath:
        model = ERFNet(NUM_CLASSES)
    elif "bisenet" in modelpath:
        model = BiSeNetV2(NUM_CLASSES)
    elif "enet" in modelpath:
        model = ENet(NUM_CLASSES)

    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
              
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                   continue
            else:
                own_state[name].copy_(param)
        return model
    
    state_dict = torch.load(weightspath, weights_only=False, map_location=lambda storage, loc: storage)

    if "erfnet" in modelpath:
        model = load_my_state_dict(model, state_dict )
    elif "bisenet" in modelpath:
        model.load_state_dict(state_dict)
    elif "enet" in modelpath:
        model.load_state_dict(state_dict["state_dict"])

    print ("Model and weights LOADED successfully")

    if args.temp != None:
        input_transform = transforms.Compose([
        transforms.Resize((512, 1024)), 
        transforms.ToTensor(),          
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        target_transform = transforms.Compose([
        transforms.Resize((512, 1024)), 
        transforms.ToTensor(), 
        transform_label_temp
        ])
        
        valid_loader = DataLoader(TestDataset(args.input[0].split("images")[0],input_transform,target_transform),
            num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
    
        model = ModelWithTemperature(model,args.temp)
        model.set_temperature(valid_loader)

    model.eval()

    start = time.time()

    for i, path in enumerate(glob.glob(os.path.expanduser(str(args.input[0])))):
        images = torch.from_numpy(np.array(Image.open(path).convert('RGB'))).unsqueeze(0).float()
        images = images.permute(0,3,1,2)
        
        if "bisenet" in modelpath:
            if "RoadAnomaly" in path:
                # Needed for bisenet in order to have dimensions multiple of 32 (cause the model downsample the images by 32)
                transform_image = transforms.Resize((704,1280))
                images = transform_image(images)
            elif "RoadObsticle" in path:
                transform_image = transforms.Resize((1056,1920))
                images = transform_image(images)

        with torch.no_grad():
            if "bisenet" in modelpath:
                result = model(images)[0]
            else:
                result = model(images)
        
        #take only background as anomaly
        background_index = 19 # background is the last one
        result_void = result[:,background_index,:,:].unsqueeze(0)

        if args.void:
            anomaly_result_void = - np.max(result_void.squeeze(0).data.cpu().numpy(), axis=0)  
            anomaly_result_full = - np.max(result.squeeze(0).data.cpu().numpy(), axis=0)  
        else:
            anomaly_result_void = 1.0 - np.max(torch.nn.functional.softmax(result_void.squeeze(0),dim=0).cpu().numpy(), axis=0)  
            anomaly_result_full = 1.0 - np.max(torch.nn.functional.softmax(result.squeeze(0),dim=0).cpu().numpy(), axis=0)  

        
        pathGT = path.replace("images", "labels_masks")                
        if "RoadObsticle21" in pathGT:
           pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
           pathGT = pathGT.replace("jpg", "png")                
        if "RoadAnomaly" in pathGT:
           pathGT = pathGT.replace("jpg", "png")  
        
        mask = Image.open(pathGT)
        if "bisenet" in modelpath:
            if "RoadAnomaly" in path:
                # Needed for bisenet in order to have dimensions multiple of 32 (cause the model downsample the images by 32)
                transform_label = transforms.Resize((704,1280))
                mask = transform_label(mask)
            elif "RoadObsticle" in path:
                transform_label = transforms.Resize((1056,1920))
                mask = transform_label(mask)

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

        # Plot comparison between void and full classifier
        if "pattern" in path:
           plot_anomaly_map(modelpath, path,ood_gts,anomaly_result_void,anomaly_result_full)  
            
        del result, anomaly_result_void,anomaly_result_full, ood_gts, mask
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

    print(f"Time for evaluation: {end-start} s")

if __name__ == '__main__':
    main()