import os
import glob
import torch
import random
from PIL import Image
import numpy as np
from erfnet import ERFNet
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import torchvision.transforms as transforms
from dataset import TestDataset
from torch.utils.data import DataLoader

seed = 42

# Riproducibilità
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

NUM_CLASSES = 20


def transform_label(label):
    return torch.squeeze(label, 0).long()


def remove_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "")  # Rimuove il prefisso "module."
        new_state_dict[name] = v
    return new_state_dict


def load_model(args):
    model = ERFNet(NUM_CLASSES)
    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print(f"Loading model: {modelpath}")
    print(f"Loading weights: {weightspath}")

    state_dict = torch.load(weightspath, map_location=lambda storage, loc: storage)
    state_dict = remove_module_prefix(state_dict)
    model.load_state_dict(state_dict, strict=False)

    if not args.cpu:
        model = torch.nn.DataParallel(model).cuda()

    model.eval()
    print("Model loaded successfully.")
    return model


def get_max_logit(output):
    # Calcolo max-logit: Differenza tra la logit più alta e la seconda più alta
    top2 = torch.topk(output, 2, dim=1)
    return top2.values[:, 0, :, :] - top2.values[:, 1, :, :]


def get_entropy(output):
    # Calcolo entropia massima
    probabilities = torch.nn.functional.softmax(output, dim=1)
    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)
    return entropy


def evaluate_ood(model, dataloader, method='max_logit'):
    anomaly_score_list = []
    ood_gts_list = []

    for images, labels in dataloader:
        images = images.cuda()
        labels = labels.numpy()

        with torch.no_grad():
            output = model(images)

        if method == 'max_logit':
            anomaly_scores = 1 - get_max_logit(output).cpu().numpy()
        elif method == 'max_entropy':
            anomaly_scores = get_entropy(output).cpu().numpy()
        else:
            raise ValueError("Invalid method. Choose 'max_logit' or 'max_entropy'.")

        anomaly_score_list.append(anomaly_scores)
        ood_gts_list.append(labels)

    ood_gts_list = np.array(ood_gts_list).flatten()

    # Controllo della presenza di campioni OOD
    unique_labels = np.unique(ood_gts_list)
    print(f"Etichette uniche nei dati: {unique_labels}")

    if 1 not in unique_labels:
        print("Attenzione: Nessun campione OOD (anomalo) trovato nel dataset!")
        print("Le metriche potrebbero non essere calcolate correttamente.")

    return np.array(anomaly_score_list), np.array(ood_gts_list)


def calculate_metrics(anomaly_scores, ood_gts):
    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]

    labels = np.concatenate([np.ones(len(ood_out)), np.zeros(len(ind_out))])
    scores = np.concatenate([ood_out, ind_out])

    auprc = average_precision_score(labels, scores)
    fpr = fpr_at_95_tpr(scores, labels)

    return auprc, fpr


def main():
    parser = ArgumentParser()
    parser.add_argument("--input", default="/path/to/dataset/images/*.jpg", nargs="+")
    parser.add_argument('--loadDir', default="trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    model = load_model(args)

    input_transform = transforms.Compose([
        transforms.Resize((512, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    target_transform = transforms.Compose([
        transforms.Resize((512, 1024)),
        transforms.ToTensor(),
        transform_label
    ])

    dataloader = DataLoader(
        TestDataset(args.input[0].split("images")[0], input_transform, target_transform),
        num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False
    )

    print("Evaluating Max-Logit...")
    logit_scores, logit_gts = evaluate_ood(model, dataloader, method='max_logit')
    logit_auprc, logit_fpr = calculate_metrics(logit_scores, logit_gts)
    print(f"Max-Logit AUPRC: {logit_auprc * 100:.2f} | FPR@95: {logit_fpr * 100:.2f}")

    print("Evaluating Max-Entropy...")
    entropy_scores, entropy_gts = evaluate_ood(model, dataloader, method='max_entropy')
    entropy_auprc, entropy_fpr = calculate_metrics(entropy_scores, entropy_gts)
    print(f"Max-Entropy AUPRC: {entropy_auprc * 100:.2f} | FPR@95: {entropy_fpr * 100:.2f}")


if __name__ == '__main__':
    main()