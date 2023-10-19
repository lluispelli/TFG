import csv
import os
import pdb
import gc

import nibabel
import torch
from monai.inferers import SimpleInferer
from monai.transforms import (
    AsDiscrete,          # need to search
    Compose,             # to apply several transformations at once
)

class Segmentation(object):

    def __init__(self, max_epochs, root_dir, device):
        self.max_epochs = max_epochs
        self.device = device
        self.root_dir = root_dir

    def build_train(self, train_loader, model, optimizer, loss_function, val_loader, val_interval, metrics,
                    training_file, val_file):

        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.val_loader = val_loader
        self.val_interval = val_interval
        self.metrics = metrics

        with open(training_file, 'w') as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=['Iteration', 'Epoch', 'Loss'])
            csv_writer.writeheader()

        with open(val_file, 'w') as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=['Epoch', 'Loss'])
            csv_writer.writeheader()

        self.training_logs = training_file
        self.val_logs = val_file

        def report_gpu():
            print(torch.cuda.list_gpu_processes())
            gc.collect()
            torch.cuda.empty_cache()

        report_gpu()

    def train(self):
        post_pred = AsDiscrete(argmax=True, dim=1)#Compose([AsDiscrete(argmax=True)])
        best_metric = -1
        best_metric_epoch = -1
        epoch_loss_values = []
        metric_values = []
        val_inferer = SimpleInferer()

        for epoch in range(self.max_epochs):
            print("-" * 10)
            print(f"epoch {epoch + 1}/{self.max_epochs}")
            self.model.train()

            epoch_loss_dict, step = self.iterate()

            to_write = [{'Iteration': it_dl,
                         'Epoch': epoch,
                         'Loss': dice_loss} for it_dl, dice_loss in enumerate(epoch_loss_dict['dice'])]

            with open(self.training_logs, 'a') as csvfile:
                csv_writer = csv.DictWriter(csvfile, fieldnames=['Iteration', 'Epoch', 'Loss'])
                csv_writer.writerows(to_write)

            mean_epoch_loss = sum(epoch_loss_dict['dice']) / step
            epoch_loss_values.append(mean_epoch_loss)
            print(f"epoch {epoch + 1} average loss: {mean_epoch_loss:.4f}")

            if (epoch + 1) % self.val_interval == 0:
                self.model.eval()
                with torch.no_grad():
                    for val_data in self.val_loader:
                        val_inputs, val_labels = self.get_input_output(val_data)
                        val_output = val_inferer(val_inputs, network=self.model)
                        self.metrics['dice'](y_pred=post_pred(val_output), y=val_labels)

                    # aggregate the final mean dice result
                    metric = self.metrics['dice'].aggregate().item()

                    with open(self.val_logs, 'a') as csvfile:
                        csv_writer = csv.DictWriter(csvfile, fieldnames=['Epoch', 'Loss'])
                        csv_writer.writerow({'Epoch': epoch, 'Loss': metric})

                    # reset the status for next validation round
                    self.metrics['dice'].reset()

                    metric_values.append(metric)
                    if metric > best_metric:
                        best_metric = metric
                        best_metric_epoch = epoch + 1
                        torch.save(self.model.state_dict(), os.path.join(self.root_dir, "best_metric_model.pth"))
                        print("saved new best metric model")
                    else:
                        torch.save(self.model.state_dict(), os.path.join(self.root_dir, "last_model.pth"))

                    print(
                        f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                        f"\nbest mean dice: {best_metric:.4f} "
                        f"at epoch: {best_metric_epoch}"
                    )
        return {'epoch_loss_values': epoch_loss_values, 'metric_values': metric_values}

    def get_input_output(self, data_dict):
        return data_dict["image"].to(self.device), data_dict["label"].to(self.device)

    def iterate(self):
        step = 0
        epoch_loss = {'dice': []}
        for batch_data in self.train_loader:

            step += 1
            inputs, labels = self.get_input_output(batch_data)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            del batch_data

            loss = self.loss_function(outputs, labels)

            del inputs, labels

            loss.backward()
            self.optimizer.step()
            epoch_loss['dice'] += [loss.item()]
            print(f"{step}/{len(self.train_loader) // self.train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")
        return epoch_loss, step


class Segmentation2Chan(Segmentation):
    def get_input_output(self, data_dict):
        return torch.cat((data_dict["image"], data_dict["vagina"]), axis=1).to(self.device), data_dict["label"].to(self.device)