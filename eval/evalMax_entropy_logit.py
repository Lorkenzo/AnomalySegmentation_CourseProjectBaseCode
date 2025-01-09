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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ERFNet(NUM_CLASSES)
    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print(f"Loading model: {modelpath}")
    print(f"Loading weights: {weightspath}")

    state_dict = torch.load(weightspath, map_location=lambda storage, loc: storage)
    state_dict = remove_module_prefix(state_dict)
    model.load_state_dict(state_dict, strict=False)

    if not args.cpu:
        model = torch.nn.DataParallel(model).to(device)

    model.eval()
    print("Model loaded successfully.")
    return model


def get_max_logit(output):
    # Calcolo max-logit: Differenza tra la logit più alta e la seconda più alta
    top2 = torch.topk(output.squeeze(0), 2, dim=0)
    return top2.values[0, :, :] - top2.values[1, :, :]


def get_entropy(output):
    # Calcolo entropia massima
    probabilities = torch.nn.functional.softmax(output.squeeze(0), dim=0)
    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=0)
    return entropy


def evaluate_ood(model, path, method='max_logit'):
    anomaly_score_list = []
    ood_gts_list = []

    for path in glob.glob(os.path.expanduser(str(path))):
        print(path)
        images = torch.from_numpy(np.array(Image.open(path).convert('RGB'))).unsqueeze(0).float()
        images = images.permute(0,3,1,2)
        with torch.no_grad():
            output = model(images)
        if method == 'max_logit':
            anomaly_scores = 1 - get_max_logit(output).cpu().numpy()
        elif method == 'max_entropy':
            anomaly_scores = get_entropy(output).cpu().numpy()
        else:
            raise ValueError("Invalid method. Choose 'max_logit' or 'max_entropy'.")

        pathGT = path.replace("images", "labels_masks")                
        if "RoadObsticle21" in pathGT:
           pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
           pathGT = pathGT.replace("jpg", "png")                
        if "RoadAnomaly" in pathGT:
           pathGT = pathGT.replace("jpg", "png")  

        mask = Image.open(pathGT)
        ood_gts = np.array(mask)

        if "RoadAnomaly" in path:
            ood_gts = np.where((ood_gts==2), 1, ood_gts)
        if "LostAndFound" in path:
            ood_gts = np.where((ood_gts==0), 255, ood_gts)
            ood_gts = np.where((ood_gts==1), 0, ood_gts)
            ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)

        if "Streethazard" in path:
            ood_gts = np.where((ood_gts==14), 255, ood_gts)
            ood_gts = np.where((ood_gts<20), 0, ood_gts)
            ood_gts = np.where((ood_gts==255), 1, ood_gts)

        if 1 not in np.unique(ood_gts):
            continue              
        else:
            ood_gts_list.append(ood_gts)
            anomaly_score_list.append(anomaly_scores)

        del output, anomaly_scores, ood_gts
        torch.cuda.empty_cache()

    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)
    return anomaly_scores, ood_gts


def calculate_metrics(anomaly_scores, ood_gts):
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

    return prc_auc, fpr


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

    print("Evaluating Max-Logit...")
    logit_scores, logit_gts = evaluate_ood(model, args.input[0],  method='max_logit')
    logit_auprc, logit_fpr = calculate_metrics(logit_scores, logit_gts)
    print(f"Max-Logit AUPRC: {logit_auprc * 100:.2f} | FPR@95: {logit_fpr * 100:.2f}")

    print("Evaluating Max-Entropy...")
    entropy_scores, entropy_gts = evaluate_ood(model, args.input[0], method='max_entropy')
    entropy_auprc, entropy_fpr = calculate_metrics(entropy_scores, entropy_gts)
    print(f"Max-Entropy AUPRC: {entropy_auprc * 100:.2f} | FPR@95: {entropy_fpr * 100:.2f}")


if __name__ == '__main__':
    main()