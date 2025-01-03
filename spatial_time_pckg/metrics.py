from . import base
from . import functional as F
from .modules import Activation
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt

class BaseMetric(base.Metric):
    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels


class IoU(base.Metric):
    __name__ = 'iou_score'

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        #y_pr = self.activation(y_pr[0]) #.squeeze(1)
        y_pr = self.activation(y_pr).squeeze(1) # without aux_params
        #print(y_pr.shape, y_pr.min(), y_pr.max())
        #print('prediction', y_pr.shape)
        #print('gt', y_gt.shape)
        return F.iou(
            y_pr, y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )

class mIoU(BaseMetric):
    __name__='miou'

    def forward(self, y_pr, y_gt):
        y_pr = torch.argmax(y_pr, dim=1).view(-1)
        y_gt = y_gt.view(-1)
        unique_classes = torch.unique(torch.cat([y_gt, y_pr]))
        iou_list = [F.iou(y_pr==cls, y_gt==cls, eps=self.eps) for cls in unique_classes]
        valid_iou = [x for x in iou_list if torch.isfinite(x)]
        return torch.mean(torch.stack(valid_iou)) if valid_iou else torch.tensor(float('nan'))

    
class mIoU2(base.Metric):
    __name__ = 'miou_score'
    
    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        
    def forward(self, y_pr, y_gt):
        #y_pr = F.softmax(pred, dim=1)
        y_pr = torch.argmax(y_pr, dim=1) #.squeeze(1)
        iou_list = list()
        present_iou_list = list()
        
        y_pr = y_pr.view(-1)
        y_gt = y_gt.view(-1)
        for sem_class in range(0, 4):
            pred_inds = (y_pr == sem_class)
            target_inds = (y_gt == sem_class)
            if target_inds.long().sum().item() == 0:
                iou_now = float('nan')
            else: 
                intersection_now = (pred_inds[target_inds]).long().sum().item()
                union_now = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection_now
                iou_now = float(intersection_now) / float(union_now)
                present_iou_list.append(iou_now)
            iou_list.append(iou_now)
            #print(round(iou_list, 2))
        return torch.as_tensor(np.mean(present_iou_list))


class Fscore(base.Metric):

    def __init__(self, beta=1,
                 eps=1e-7,
                 threshold=0.5,
                 activation=None,
                 ignore_channels=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        #y_pr = self.activation(y_pr[0])
        y_pr = self.activation(y_pr).squeeze(1)

        
        return F.f_score(
            y_pr, y_gt,
            eps=self.eps,
            beta=self.beta,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Accuracy(base.Metric):

    def __init__(self, threshold=0.5,
                 activation=None,
                 ignore_channels=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.accuracy(
            y_pr, y_gt,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Recall(base.Metric):

    def __init__(self, eps=1e-7,
                 threshold=0.5,
                 activation=None,
                 ignore_channels=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        #y_pr = self.activation(y_pr[0])
        y_pr = self.activation(y_pr) #without aux_params
        return F.recall(
            y_pr, y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Precision(base.Metric):

    def __init__(self, eps=1e-7,
                 threshold=0.5,
                 activation=None,
                 ignore_channels=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        #y_pr = self.activation(y_pr[0])
        y_pr = self.activation(y_pr)
        return F.precision(
            y_pr, y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


# Function to calculate evaluation metrics
def calculate_metrics(predictions, labels):
    # Flatten the tensors to make them 1D (for easy computation)
    predictions = predictions.view(-1)
    labels = labels.view(-1)
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()
    # Calculate True Positives, False Positives, False Negatives, True Negatives
    TP = (predictions * labels).sum().item()
    FP = (predictions * (1 - labels)).sum().item()
    FN = ((1 - predictions) * labels).sum().item()
    TN = ((1 - predictions) * (1 - labels)).sum().item()
    
    # IoU
    intersection = TP
    union = TP + FP + FN
    iou = intersection / union if union != 0 else 0

    # Precision
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0

    # Recall
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0

    # F-score
    f_score = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    # Accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    return iou, precision, recall, f_score, accuracy


# Function to evaluate model on test loader
def evaluate_model(model, test_loader, device):
    model.eval()  # Set model to evaluation mode
    iou_total, precision_total, recall_total, f_score_total, accuracy_total = 0, 0, 0, 0, 0
    num_batches = 0

    with torch.no_grad():  # Disable gradient calculation for faster computation
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Get model predictions
            outputs = model(images)
            outputs = torch.nn.functional.interpolate(
                outputs,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False)
            # print("Max Value: ", outputs.max().cpu().numpy())
            # print("Min Value: ", outputs.min().cpu().numpy())
            predictions = outputs > 0.5  # Binarize predictions
            
            # Calculate metrics for the batch
            iou, precision, recall, f_score, accuracy = calculate_metrics(predictions, labels)
            
            # Accumulate metrics
            iou_total += iou
            precision_total += precision
            recall_total += recall
            f_score_total += f_score
            accuracy_total += accuracy
            num_batches += 1

    # Calculate average metrics
    avg_iou = iou_total / num_batches
    avg_precision = precision_total / num_batches
    avg_recall = recall_total / num_batches
    avg_f_score = f_score_total / num_batches
    avg_accuracy = accuracy_total / num_batches

    return {
        'IoU': avg_iou,
        'Precision': avg_precision,
        'Recall': avg_recall,
        'F-score': avg_f_score,
        'Accuracy': avg_accuracy
    }


def visualize_predictions(model, data_loader, device, num_images=5, figsize=(15, 5), binary=False, threshold=0.5):
    """
    Visualize input images, model predictions, and ground truth labels side by side.
    
    Parameters:
        model (torch.nn.Module): Trained model for generating predictions.
        data_loader (torch.utils.data.DataLoader): DataLoader containing validation or test data.
        device (torch.device): Device on which the model and data are running (e.g., 'cuda' or 'cpu').
        num_images (int): Number of images to display. Default is 5.
        figsize (tuple): Size of the figure for displaying images. Default is (15, 5).
        binary (bool): If True, handles binary segmentation. If False, handles multiclass segmentation. Default is False.
        threshold (float): Threshold for binary segmentation. Default is 0.5.
    """
    model.eval()  # Set the model to evaluation mode
    actual_img_count = 0

    # No need for gradient calculations
    with torch.no_grad():
        for inp, lab in data_loader:
            inp, lab = inp.to(device).detach(), lab.to(device).detach()
            pred = model(inp)#.detach()
            pred = torch.nn.functional.interpolate(
                    pred,
                    size=lab.shape[-2:],
                    mode="bilinear",
                    align_corners=False)

            current_batch_size = len(inp)

            for i in range(current_batch_size):
                # Convert input, prediction, and ground truth to numpy arrays for visualization
                lab_unit = lab[i].cpu().numpy()
                inp_unit = np.transpose(inp[i].cpu().numpy(), (1, 2, 0))  # Convert input to (H, W, C) format

                # Handling binary and multiclass segmentation
                if binary:
                    # Binary: Apply threshold to convert probabilities into binary mask (0 or 1)
                    pred_img = pred[i].cpu().numpy()[0]  # Single-channel output for binary segmentation
                    pred_img = (pred_img > threshold).astype(np.uint8)  # Threshold to get binary mask
                else:
                    # Multiclass: Use argmax to get the class with the highest probability
                    pred_img = pred[i].cpu().numpy()
                    pred_img = np.argmax(pred_img, 0)

                # Plot the original image, prediction, and ground truth
                f, ax = plt.subplots(1, 3, figsize=figsize)

                # Visualization
                ax[0].imshow(inp_unit[:, :, 0])  # Assuming the input is normalized, visualize only one channel
                ax[1].imshow(pred_img)
                ax[2].imshow(lab_unit)

                # Set titles
                ax[0].set_title(f'Original Image | {actual_img_count + 1}')
                ax[1].set_title(f'Prediction | {actual_img_count + 1}')
                ax[2].set_title(f'Ground Truth | {actual_img_count + 1}')

                # Remove axis ticks
                for a in ax:
                    a.set_xticks([])
                    a.set_yticks([])

                plt.tight_layout()
                plt.show()

                actual_img_count += 1

                # Stop after displaying the specified number of images
                if actual_img_count >= num_images:
                    return