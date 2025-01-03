from . import base
from . import functional as F
from ..base.modules import Activation
import torch
import numpy as np


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
    __name__ = 'miou'

    def forward(self, y_pr, y_gt):
        # Ensure proper shapes
        y_pr = torch.argmax(y_pr, dim=1)  # Convert predictions to class indices
        y_pr = y_pr.view(-1)              # Flatten predictions
        y_gt = y_gt.view(-1)              # Flatten ground truth

        # Compute IoU for each unique class
        unique_classes = torch.unique(y_gt)
        iou_list = [compute_iou(y_pr == cls, y_gt == cls, eps=1e-6) for cls in unique_classes]

        # Filter valid IoU values
        valid_iou = [x for x in iou_list if torch.isfinite(x)]

        # Return mean IoU or NaN if no valid IoU
        return torch.mean(torch.stack(valid_iou)) if valid_iou else torch.tensor(float('nan'))



# Helper function for IoU calculation
def compute_iou(y_pr, y_gt, eps=1e-6):
    intersection = torch.sum((y_pr & y_gt).float())
    union = torch.sum((y_pr | y_gt).float())
    iou = (intersection + eps) / (union + eps)
    return iou



    
class mIoU2(base.Metric):
    __name__ = 'miou_score'
    
    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        """
        Calculate mIoU for batch predictions.
        Args:
            y_pr: Predicted tensor of shape (batch_size, num_classes, height, width).
            y_gt: Ground truth tensor of shape (batch_size, height, width).
        """
        # Convert predictions to class indices
        # y_pr = torch.argmax(y_pr, dim=0)  # Shape: (batch_size, height, width)

        # Initialize lists to store IoU for each class
        iou_list = []
        present_iou_list = []

        batch_size = y_pr.size(0)  # Batch size

        # Compute IoU for each sample in the batch
        for i in range(batch_size):
            sample_pred = y_pr[i]  # Shape: (height, width)
            sample_gt = y_gt[i]    # Shape: (height, width)

            # Flatten the tensors for comparison
            sample_pred = sample_pred.view(-1)  # Shape: (height * width,)
            sample_gt = sample_gt.view(-1)      # Shape: (height * width,)

            for sem_class in range(4):  # Adjust for your number of classes
                pred_inds = (sample_pred == sem_class)  # Shape: (height * width,)
                target_inds = (sample_gt == sem_class)  # Shape: (height * width,)

                # Check if the class exists in the ground truth
                if target_inds.sum().item() == 0:
                    iou_now = float('nan')  # No ground truth for this class
                else:
                    # Calculate intersection and union
                    intersection_now = (pred_inds & target_inds).sum().item()
                    union_now = pred_inds.sum().item() + target_inds.sum().item() - intersection_now
                    iou_now = float(intersection_now) / float(union_now)
                    present_iou_list.append(iou_now)

                iou_list.append(iou_now)

        # Compute mean IoU over all present classes in the batch
        return torch.tensor(np.nanmean(present_iou_list))  # Use nanmean to ignore NaNs






class Fscore(base.Metric):

    def __init__(self, beta=1, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
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

    def __init__(self, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
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

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
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

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
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
