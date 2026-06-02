import os
import csv

import torch
from torchvision import datasets

from model import build_model
from preprocess import preprocess_data


def predict(weights_path='model.pth', out_path='submission.csv'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # we load our own checkpoint over it and replace weights
    model = build_model(num_classes=100, pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()

    # ImageFolder sorts class folder names as strings ('0','1',...),
    # so the model's output index i corresponds to class_names[i] NOT to int i
    class_names = datasets.ImageFolder(os.path.join('data', 'train')).classes

    # preprocess_data() builds the 'test' loader with the same 384px eval transform;
    # it yields (image_tensor, file_path) batches.
    dataloaders, _ = preprocess_data()
    test_loader = dataloaders['test']

    rows = []
    with torch.no_grad():
        for inputs, paths in test_loader:
            inputs = inputs.to(device)
            preds = model(inputs).argmax(1)
            for path, pred in zip(paths, preds):
                rows.append((os.path.basename(path), class_names[pred.item()]))

    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'Label'])
        writer.writerows(rows)

    print(f'Wrote {len(rows)} predictions to {out_path}')


if __name__ == '__main__':
    predict()
