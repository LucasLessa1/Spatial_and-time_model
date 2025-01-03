import sys
import torch
from tqdm import tqdm as tqdm
from .meter import AverageValueMeter
import numpy as np


class Epoch:

    def __init__(self, model, loss, metrics, stage_name, device='cpu', verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for x, y in iterator:
                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x, y)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__class__.__name__: loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class TrainEpoch(Epoch):

    def __init__(self, model, loss, metrics, optimizer, scheduler=None, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='train',
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer
        self.scheduler = scheduler

    def on_epoch_start(self):
        self.model.train()

    # def batch_update(self, x, y):
    #     self.optimizer.zero_grad()
    #     prediction = self.model.forward(x)
    #     loss = self.loss(prediction, y)
    #     loss.backward()
    #     self.optimizer.step()
    #     return loss, prediction
    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        prediction = self.model.forward(x)
        prediction = torch.nn.functional.interpolate(
            prediction,
            size=y.shape[-2:],
            mode="bilinear",
            align_corners=False)
        # Usando outra loss eu preciso
        prediction = prediction.squeeze(1)
        # prediction = torch.argmax(prediction, dim=1)
        # y = y[:, 1, :, :]
        if self.loss.__class__.__name__ == 'CrossEntropyLoss':
            y = y.long()
        else:
            y = y.float()
        # prediction = self.convert_prediction_to_labels(prediction)
        # y = y.float()
        loss = self.loss(prediction, y)
        loss.backward()
        self.optimizer.step()
        return loss, prediction

    def convert_prediction_to_labels(self, prediction):
        rgb_to_label = {
        (64, 128, 64): 0,  # Animal
        (192, 0, 128): 1,  # Archway
        (0, 128, 192): 2,  # Bicyclist
        (0, 128, 64): 3,   # Bridge
        (128, 0, 0): 4,    # Building
        (64, 0, 128): 5,   # Car
        (64, 0, 192): 6,   # CartLuggagePram
        (192, 128, 64): 7, # Child
        (192, 192, 128): 8,# Column_Pole
        (64, 64, 128): 9,  # Fence
        (128, 0, 192): 10, # LaneMkgsDriv
        (192, 0, 64): 11,  # LaneMkgsNonDriv
        (128, 128, 64): 12,# Misc_Text
        (192, 0, 192): 13, # MotorcycleScooter
        (128, 64, 64): 14, # OtherMoving
        (64, 192, 128): 15,# ParkingBlock
        (64, 64, 0): 16,   # Pedestrian
        (128, 64, 128): 17,# Road
        (128, 128, 192): 18,# RoadShoulder
        (0, 0, 192): 19,   # Sidewalk
        (192, 128, 128): 20,# SignSymbol
        (128, 128, 128): 21,# Sky
        (64, 128, 192): 22,# SUVPickupTruck
        (0, 0, 64): 23,    # TrafficCone
        (0, 64, 64): 24,   # TrafficLight
        (192, 64, 128): 25,# Train
        (128, 128, 0): 26, # Tree
        (192, 128, 192): 27,# Truck_Bus
        (64, 0, 64): 28,   # Tunnel
        (192, 192, 0): 29, # VegetationMisc
        (0, 0, 0): 30,     # Void
        (64, 192, 0): 31   # Wall
    }
        prediction_np = prediction.detach().cpu().numpy()

        prediction_labels = np.zeros(prediction_np.shape[:2], dtype=np.uint8)

        for rgb, label in rgb_to_label.items():
            mask = np.all(prediction_np == rgb, axis=-1)
            prediction_labels[mask] = label

        return torch.tensor(prediction_labels, dtype=torch.long).to(prediction.device)



class ValidEpoch(Epoch):

    def __init__(self, model, loss, metrics, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='valid',
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            prediction = self.model.forward(x)
            prediction = torch.nn.functional.interpolate(
                prediction,
                size=y.shape[1:3],
                mode="bilinear",
                align_corners=False)
            # Usando outra loss eu preciso
            prediction = prediction.squeeze(1)
            if self.loss.__class__.__name__ == 'CrossEntropyLoss':
                y = y.long()
            else:
                y = y.float()
            # y = y.float()
            loss = self.loss(prediction, y)
        return loss, prediction


class TestEpoch(Epoch):

    def __init__(self, model, loss, metrics, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='test',
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            prediction = self.model.forward(x)
            loss = self.loss(prediction, y)
        return loss, prediction
