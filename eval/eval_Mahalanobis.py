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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    """
    Calcola la media µ_c per ogni classe c.
    Args:
        features (ndarray): matrice di forma (N, D)
        labels (ndarray): vettore di etichette di forma (N,)
    Returns:
        dict: chiavi = classi c, valori = µ_c (ndarray di forma (D,))
    """
    class_means = {}
    classes = np.unique(labels)
    for c in classes:
        class_features = features[labels == c]
        mu_c = np.mean(class_features, axis=0)
        class_means[c] = mu_c
    return class_means


def compute_covariance(features, labels, class_means):
    """
    Calcola la matrice di covarianza complessiva.
    Args:
        features (ndarray): matrice di forma (N, D)
        labels (ndarray): vettore di etichette di forma (N,)
        class_means (dict): dizionario µ_c per ciascuna classe
    Returns:
        ndarray: matrice di covarianza di forma (D, D)
    """
    N = features.shape[0]
    D = features.shape[1]
    covariance = np.zeros((D, D))

    for i in range(N):
        x_i = features[i]
        y_i = labels[i]
        mu_c = class_means[y_i]
        diff = (x_i - mu_c).reshape(-1, 1)
        covariance += diff @ diff.T

    return covariance / N

def mahalanobis_distance(x, mu, inv_cov):
    """
    Calcola la distanza di Mahalanobis tra x e mu con matrice di covarianza inversa.
    """
    diff = x - mu
    return diff.T @ inv_cov @ diff


def compute_mahalanobis_confidence(x, model, feature_extractor, class_means, inv_covs, alpha, epsilon):
    """
    Calcola il punteggio di confidenza basato sulla distanza di Mahalanobis.

    Args:
        x (ndarray): input test sample (numpy array o torch, dipende dal framework usato)
        model: il modello completo (es. rete neurale)
        feature_extractor: funzione che estrae feature da tutti i layer (ritorna lista f_l(x))
        class_means (dict): dizionario {layer -> {class_label -> µ_l,c}}
        inv_covs (dict): dizionario {layer -> inv(Σ_l)}
        alpha (dict): pesi alpha_l della combinazione lineare
        epsilon (float): valore del rumore per perturbare l'input

    Returns:
        float: confidence score finale
    """
    M = {}  # M_l per ogni layer

    features = feature_extractor(x)

    for l, f_l_x in features.items():
        mu_l = class_means[l]
        inv_cov_l = inv_covs[l]

        # Trova la classe più vicina
        distances = {c: mahalanobis_distance(f_l_x, mu_l[c], inv_cov_l) for c in mu_l}
        closest_class = min(distances, key=distances.get)
        mu_closest = mu_l[closest_class]

        # Calcola gradiente per perturbazione (simulata: qui semplificata)
        # In pratica, dovresti usare backprop per ∇ₓ Mahalanobis(f_l(x), µ_l,c)
        # Qui semplifichiamo con il gradiente del L2 solo per esempio
        grad_sim = np.sign(f_l_x - mu_closest)

        # Perturba x (qui supponiamo sia nello spazio feature già, altrimenti servirebbe grad reale)
        f_l_x_perturbed = f_l_x - epsilon * grad_sim

        # Calcola punteggio massimo di Mahalanobis con x perturbato
        perturbed_distances = {
            c: mahalanobis_distance(f_l_x_perturbed, mu_l[c], inv_cov_l)
            for c in mu_l
        }
        M[l] = -min(perturbed_distances.values())  # Negativo perché score = max( -dist )

    # Calcolo della confidence finale con pesi α_l
    confidence = sum(alpha[l] * M[l] for l in M)
    return confidence

def main(args):
    model = ERFNet(NUM_CLASSES)
    weightspath = args.loadDir + args.loadWeights

    if not args.cpu:
        model = torch.nn.DataParallel(model).cuda()

    state_dict = torch.load(weightspath, weights_only=True, map_location=lambda storage, loc: storage)
    state_dict = remove_module_prefix(state_dict)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

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

    selected_layers = ['encoder.layers.0.downsample']
    feature_extractor = get_feature_extractor(model, selected_layers.copy())

    all_features = []
    all_labels = []

    for step, (images, labels, _, _) in enumerate(loader):
        if not args.cpu:
            images = images.cuda()
            labels = labels.cuda()

        feats = feature_extractor(images)
        for layer_name, feat in feats.items():
            all_features.append(feat.reshape(-1, feat.shape[-1]))
            all_labels.append(labels.cpu().numpy().reshape(-1))
        if step > 20:
            break

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    class_means = {selected_layers[0]: compute_class_means(all_features, all_labels)}
    cov = compute_covariance(all_features, all_labels, class_means[selected_layers[0]])
    inv_covs = {selected_layers[0]: np.linalg.inv(cov)}
    alpha = {selected_layers[0]: 1.0}

    all_scores = []

    for step, (images, labels, filename, _) in enumerate(loader):
        if not args.cpu:
            images = images.cuda()
        confidence_score = compute_mahalanobis_confidence(
            images, model, feature_extractor, class_means, inv_covs, alpha, epsilon=0.01
        )
        print(f"{step}: {filename[0]} -> Mahalanobis confidence score: {confidence_score:.4f}")
        all_scores.append((filename[0], confidence_score))

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
