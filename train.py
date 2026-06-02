import os
import time
from tempfile import TemporaryDirectory

import torch


# Training loop adapted from the PyTorch transfer-learning tutorial.
# batch_mix, if given, is a torchvision v2 MixUp/CutMix transform applied to
# each training batch (it returns mixed images + soft labels).
def train_model(model, criterion, optimizer, scheduler, num_epochs=25, dataset_sizes=None, dataloaders=None, device=None, batch_mix=None):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    # MixUp/CutMix (train phase only) -> soft labels [B, num_classes]
                    use_mix = phase == 'train' and batch_mix is not None
                    with torch.set_grad_enabled(phase == 'train'):
                        if use_mix:
                            mixed, soft_labels = batch_mix(inputs, labels)
                            outputs = model(mixed)
                            loss = criterion(outputs, soft_labels)
                        else:
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    if use_mix:
                        # train acc vs the dominant (argmax) class of the mixed label
                        running_corrects += torch.sum(preds == soft_labels.argmax(1))
                    else:
                        running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:.4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
    return model

