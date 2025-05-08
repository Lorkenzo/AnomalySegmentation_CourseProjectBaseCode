# Code to calculate IoU (mean and per-class) in a dataset
# Nov 2017
# Eduardo Romera
#######################

import numpy as np
import torch
import torch.nn.functional as F
import os
import importlib
import time
import torchvision.transforms as transforms
from PIL import Image
from argparse import ArgumentParser
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage
from dataset import cityscapes
from models.erfnet import ERFNet
from models.erfnetQ import ERFNetQ
from temperature_scaling import ModelWithTemperature
from transform import Relabel, ToLabel, Colorize
from iouEval import iouEval, getColorEntry

from torch.utils.data import random_split

from torch.quantization.observer import MinMaxObserver,HistogramObserver
from torchinfo import summary
import torch.nn.utils.prune as prune

NUM_CHANNELS = 3
NUM_CLASSES = 20

image_transform = ToPILImage()
input_transform_cityscapes = Compose([
    Resize(512, Image.BILINEAR),
    ToTensor(),
])
target_transform_cityscapes = Compose([
    Resize(512, Image.NEAREST),
    ToLabel(),
    Relabel(255, 19),   #ignore label to 19
])

def compute_model_stats(model, input_size):
    summary(model, input_size=input_size, col_names=["input_size", "output_size", "num_params", "mult_adds"], depth=0)

def apply_pruning(model,image_size, amount=0.2):
    
    # Convolutional layers are the ones more suitable for pruning
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.ConvTranspose2d):
            prune.l1_unstructured(module, name='weight',amount=amount)  # Prune the weights

    #compute_model_stats(model, image_size)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.ConvTranspose2d):
            prune.remove(module, 'weight')
    
    return model
   
def main(args):

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    if args.quantize:
        model = ERFNetQ(NUM_CLASSES)
    else: 
        model = ERFNet(NUM_CLASSES)

    #model = torch.nn.DataParallel(model)
    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath, weights_only=True, map_location=lambda storage, loc: storage))
    print ("Model and weights LOADED successfully")

    if(not os.path.exists(args.datadir)):
        print ("Error: datadir could not be loaded")


    full_dataset = cityscapes(args.datadir, input_transform_cityscapes, target_transform_cityscapes, subset=args.subset)

    calib_loader = DataLoader(full_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    image, _, _, _ = next(iter(calib_loader))
    image_size = image.shape # (H, W)
    
    # Pruning the model 

    model = apply_pruning(model,image_size, amount=args.prune)

    model.eval()

    if (torch.cuda.is_available() and not args.cpu):
        model = model.cuda()

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

    # Prepare for quantization
    model = torch.quantization.prepare(model, inplace=False)
    
    # Calibration
    for step, (images, labels, _, _) in enumerate(calib_loader):
        if (torch.cuda.is_available() and not args.cpu):
            images = images.cuda()
            labels = labels.cuda()

        inputs = Variable(images)
        with torch.no_grad():
            _ = model(inputs)
        
    #Convert to quantized model
    if (torch.cuda.is_available() and not args.cpu):
        model.cpu()
    
    model = torch.quantization.convert(model, inplace=False) 
    # Check new stats of the model
    print("\n\t\tComputing model stats after quantization...")
    #compute_model_stats(model, image_size)

    end = time.time()

    print(f"Time for quantization: {end-start} s")
    torch.save(model.state_dict(),f"./trained_models/erfnet_quantized_{int(args.prune*100)}.pth")
    print(f"model saved to ./trained_models/erfnet_quantized_{int(args.prune*100)}.pth")
    
if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--state')

    parser.add_argument('--loadDir',default="trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--prune', type=float, default=0.2)

    main(parser.parse_args())
