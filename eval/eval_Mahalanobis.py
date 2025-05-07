import numpy as np
import torch
import torch.nn.functional as F
import os
import importlib
import time
import cityscapesscripts as cs
import torchvision.transforms as transforms

from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage

from dataset import cityscapes
from models.erfnet import ERFNet
from temperature_scaling import ModelWithTemperature
from transform import Relabel, ToLabel, Colorize
from iouEval import iouEval, getColorEntry
import random


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

def transform_label(label):
    return torch.squeeze(label, 0).long()


def remove_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "")  # Rimuove il prefisso "module."
        new_state_dict[name] = v
    return new_state_dict

def load_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = ERFNet(NUM_CLASSES)
    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print(f"Loading model: {modelpath}")
    print(f"Loading weights: {weightspath}")

    state_dict = torch.load(weightspath, weights_only=True, map_location=lambda storage, loc: storage)
    state_dict = remove_module_prefix(state_dict)
    model.load_state_dict(state_dict, strict=False)

    if not args.cpu:
        model = torch.nn.DataParallel(model).to(device)

    model.eval()
    print("Model loaded successfully.")
    return model

def get_feature_extractor(model, layers):
    features = {}
    layers_copy = layers.copy()

    def hook_gen(layer_name):
        def hook(module, input, output):
            features[layer_name] = output.detach().cpu().numpy().squeeze()
        return hook

    for name, module in model.named_modules():
        if name in layers_copy:
            module.register_forward_hook(hook_gen(name))

    def extractor(x):
        features.clear()
        model(x)
        return features.copy()

    return extractor

def compute_class_means(features, labels):
    print(f"Shape of features inside compute_class_means: {features.shape}")
    print(f"Shape of labels inside compute_class_means: {labels.shape}")
    class_means = {}
    classes = np.unique(labels)
    for c in classes:
        class_features = features[labels == c]
        if class_features.size > 0:
            mu_c = np.mean(class_features, axis=0)
            class_means[c] = mu_c
    return class_means


def compute_covariance(features, labels, class_means):
    N = features.shape[0]
    D = features.shape[1]
    covariance = np.zeros((D, D))

    for i in range(N):
        x_i = features[i]
        y_i = labels[i]
        if y_i in class_means:
            mu_c = class_means[y_i]
            diff = (x_i - mu_c).reshape(-1, 1)
            covariance += diff @ diff.T

    return covariance / N if N > 0 else np.eye(D)

def mahalanobis_distance(x, mu, inv_cov):
    diff = x - mu
    return diff.T @ inv_cov @ diff

def compute_mahalanobis_confidence(x, model, feature_extractor, class_means, inv_covs, alpha, epsilon):
    M = {}  # M_l per ogni layer

    features = feature_extractor(x)

    for l, feat_map in features.items():
        if l in class_means and l in inv_covs and l in alpha:
            mu_l = class_means[l]
            inv_cov_l = inv_covs[l]

            feat_shape = feat_map.shape
            if len(feat_shape) == 3:
                num_channels, height, width = feat_shape
                batch_size = 1
            elif len(feat_shape) == 4:
                batch_size, num_channels, height, width = feat_shape
            else:
                print(f"Unexpected feature map shape: {feat_shape}")
                continue

            # Flatten the spatial dimensions of the feature map
            flattened_feat_map = feat_map.reshape(batch_size, num_channels, -1).transpose(0, 2, 1).reshape(-1, num_channels)

            distances = {}
            for c in mu_l:
                # Calculate Mahalanobis distance for each feature vector to the class mean
                dist_to_class = np.array([mahalanobis_distance(f, mu_l[c], inv_cov_l) for f in flattened_feat_map])
                distances[c] = np.mean(dist_to_class) # Or perhaps min/max depending on your anomaly definition

            if distances:
                closest_class = min(distances, key=distances.get)
                mu_closest = mu_l[closest_class]

                # Flatten the feature map for perturbation as well
                grad_sim = np.sign(flattened_feat_map - mu_closest)
                f_l_x_perturbed = flattened_feat_map - epsilon * grad_sim

                perturbed_distances = {}
                for c in mu_l:
                    perturbed_dist_to_class = np.array([mahalanobis_distance(f, mu_l[c], inv_cov_l) for f in f_l_x_perturbed])
                    perturbed_distances[c] = np.mean(perturbed_dist_to_class)

                if perturbed_distances:
                    M[l] = -min(perturbed_distances.values())  # Negativo perché score = max( -dist )

    confidence = sum(alpha.get(l, 0) * M.get(l, 0) for l in alpha)
    return confidence

def main(args):
    model = load_model(args)

    input_transform = transforms.Compose([
        transforms.Resize((128, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    target_transform = transforms.Compose([
        transforms.Resize((128, 512)),
        transforms.ToTensor(),
    ])

    loader = DataLoader(cityscapes(args.datadir, input_transform, target_transform, subset=args.subset),
                        num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    # ********************** CORRECTION 1: Choose the layer **********************
    selected_layer_name = 'module.decoder.output_conv'
    # ****************************************************************************

    feature_extractor = get_feature_extractor(model, [selected_layer_name])

    all_features = []
    all_labels = []

    print(f"Extracting features from layer: {selected_layer_name} for mean and covariance calculation...")
    for step, (images, labels, _, _) in enumerate(loader):
        if not args.cpu and torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        feats = feature_extractor(images)
        for layer_name, feat in feats.items():
            if layer_name == selected_layer_name:
                # ********************** CORRECTION 2: Reshape features (without permute) **********************
                feat_shape = feat.shape
                if len(feat_shape) == 4:
                    batch_size, num_channels, height, width = feat_shape
                    reshaped_feat = feat.reshape(batch_size, num_channels, -1).transpose(0, 2, 1).reshape(-1, num_channels)
                else:
                    # Assuming shape is (C, H, W) - reshape to (H*W, C)
                    num_channels, height, width = feat_shape
                    reshaped_feat = feat.reshape(num_channels, -1).transpose(1, 0)
                # ****************************************************************************
                flattened_labels = labels.cpu().numpy().reshape(-1)

                all_features.append(reshaped_feat)
                all_labels.append(flattened_labels)

        print(f"Processed batch {step + 1} for feature extraction.")
        if step > 20:
            break

    print(f"Length of all_features: {len(all_features)}")
    print(f"Length of all_labels: {len(all_labels)}")

    if not all_features:
        print("Error: all_features is empty. Check data loading and feature extraction.")
        return
    if not all_labels:
        print("Error: all_labels is empty. Check data loading and label handling.")
        return

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    print(f"Shape of all_features after concatenation: {all_features.shape}")
    print(f"Shape of all_labels after concatenation: {all_labels.shape}")

    class_means = {selected_layer_name: compute_class_means(all_features, all_labels)}
    cov = compute_covariance(all_features, all_labels, class_means[selected_layer_name])
    try:
        inv_covs = {selected_layer_name: np.linalg.inv(cov)}
    except np.linalg.LinAlgError:
        print("Error: Singular covariance matrix. Cannot compute inverse.")
        return
    alpha = {selected_layer_name: 1.0}

    all_scores = []

    print("Computing Mahalanobis confidence scores...")
    loader_eval = DataLoader(cityscapes(args.datadir, input_transform, target_transform, subset=args.subset),
                             num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    feature_extractor_eval = get_feature_extractor(model, [selected_layer_name])

    for step, (images, labels, filename, _) in enumerate(loader_eval):
        if not args.cpu and torch.cuda.is_available():
            images = images.cuda()
        confidence_score = compute_mahalanobis_confidence(
            images, model, feature_extractor_eval, class_means, inv_covs, alpha, epsilon=0.01
        )
        print(f"{step}: {filename} -> Mahalanobis confidence score: {confidence_score:.4f}")
        all_scores.append((filename, confidence_score))

    with open("mahalanobis_scores.txt", "w") as f:
        for fname, score in all_scores:
            f.write(f"{fname}\t{score:.4f}\n")

    print("Valutazione Mahalanobis completata.")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--loadDir', default="trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    main(parser.parse_args())