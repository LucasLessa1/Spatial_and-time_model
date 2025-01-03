import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class CNNLSTMModel(nn.Module):
    def __init__(self, input_channels=11, spatial_dim=(256, 256), cnn_out_channels=128, lstm_hidden_size=64, num_classes=1):
        super(CNNLSTMModel, self).__init__()

        # CNN backbone to extract spatial features
        self.cnn = nn.Sequential(
            nn.Conv2d(1, cnn_out_channels, kernel_size=3, padding=1),  # Cada time step tem 1 canal
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Reduz as dimensões espaciais
        )

        # Calcula as dimensões espaciais reduzidas após a CNN
        reduced_h, reduced_w = spatial_dim[0] // 2, spatial_dim[1] // 2
        self.reduced_dim = reduced_h * reduced_w
        self.cnn_out_channels = cnn_out_channels

        # LSTM para modelagem temporal
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,  # Cada time_step é representado pelos canais da CNN
            hidden_size=lstm_hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # Camada totalmente conectada para mapear a saída do LSTM para a segmentação final
        self.fc = nn.Conv2d(
            in_channels=lstm_hidden_size * 2,  # O LSTM bidirecional dobra o tamanho do hidden
            out_channels=num_classes,
            kernel_size=1
        )

    def forward(self, x):
        batch_size, time_steps, height, width = x.shape

        # Prepara o batch e os time_steps para a CNN
        x = x.view(batch_size * time_steps, 1, height, width)

        # Extrai recursos espaciais usando a CNN
        spatial_features = self.cnn(x)  # Shape: [B*T, C, H', W']
        _, channels, h, w = spatial_features.shape

        # Reorganiza os dados para o LSTM: [batch_size * height * width, time_steps, channels]
        spatial_features = spatial_features.view(batch_size, time_steps, channels, h, w)
        spatial_features = spatial_features.permute(0, 3, 4, 1, 2)  # [B, H', W', T, C]
        spatial_features = spatial_features.reshape(batch_size * h * w, time_steps, channels)

        # Passa pelo LSTM
        lstm_out, _ = self.lstm(spatial_features)  # Shape: [B*H*W, T, 2*Hid]

        # Pega a última saída do LSTM
        lstm_out = lstm_out[:, -1, :]  # Shape: [B*H*W, 2*Hid]

        # Reorganiza de volta para o formato espacial
        lstm_out = lstm_out.view(batch_size, h, w, -1).permute(0, 3, 1, 2)  # [B, 2*Hid, H', W']

        # Saída final de segmentação
        output = self.fc(lstm_out)  # Shape: [B, Num_Classes, H', W']

        return output
