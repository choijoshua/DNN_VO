import torch
import torch.nn as nn


class ConvLayer(nn.Module):

    def __init__(self, input_channels=6, input_img_size=(752, 480), conv_dropout=0.1):
        
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=conv_dropout)
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=7, padding=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, padding=2, stride=2)
        self.conv3_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=2)
        self.conv4_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=2)
        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1, stride=2)
        self.flatten = nn.Flatten()

        x = torch.zeros((1, input_channels, input_img_size[0], input_img_size[1]))
        self.dim = self.forward(x).size()[-1]

    def forward(self, x):

        x = self.dropout(self.relu(self.conv1(x)))
        x = self.dropout(self.relu(self.conv2(x)))
        x = self.dropout(self.relu(self.conv3(x)))
        x = self.dropout(self.relu(self.conv3_1(x)))
        x = self.dropout(self.relu(self.conv4(x)))
        x = self.dropout(self.relu(self.conv4_1(x)))
        x = self.dropout(self.relu(self.conv5(x)))
        x = self.dropout(self.relu(self.conv5_1(x)))
        x = self.dropout(self.conv6(x))
        x = self.flatten(x)
        return x

class DeepVO(nn.Module):

    def __init__(self, input_channels=6, image_size=(192, 640), hidden_size=1000, lstm_layers=2, num_classes=6, lstm_dropout=0.2, conv_dropout=0.1, batch_size=4):

        super().__init__()
        self.feature_extractor = ConvLayer(input_channels=input_channels, 
                                           input_img_size=image_size, 
                                           conv_dropout=conv_dropout)
        
        self.lstm = nn.LSTM(
            input_size=self.feature_extractor.dim,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout
        )

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        features = self.feature_extractor(x)
        features = torch.unsqueeze(features, dim=1)
        lstm_out, self.hidden_state = self.lstm(features)

        y = self.fc(lstm_out)
        
        return y
    


