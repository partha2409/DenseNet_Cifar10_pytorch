import torch.nn as nn
import torch
import torch.nn.functional as F


class DenseLayer(nn.Module):

    def __init__(self, in_features, growth_rate):
        super(DenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(num_features=in_features)
        self.conv_1x1 = nn.Conv2d(in_channels=in_features, out_channels=growth_rate, kernel_size=1, stride=1)

        self.bn2 = nn.BatchNorm2d(num_features=in_features+growth_rate)
        self.conv_3x3 = nn.Conv2d(in_channels=in_features+growth_rate, out_channels=growth_rate, kernel_size=1, stride=1)

    def forward(self, x):
        temp = x
        out = F.relu(self.bn1(x))
        out = self.conv_1x1(out)
        temp1 = torch.cat((out, temp), 1)
        out = F.relu(self.bn2(temp1))
        out = self.conv_3x3(out)
        out = torch.cat((out, temp1), 1)
        return out


class DenseBlock(nn.Module):

    def __init__(self, growth_rate, n_layers):
        super(DenseBlock, self).__init__()
        self.dense_layers = nn.ModuleList()
        self.n_layers = n_layers
        for i in range(n_layers):
            self.in_features = growth_rate + (growth_rate * i * 2)
            self.dense_layers.append(DenseLayer(in_features=self.in_features, growth_rate=growth_rate))

    def forward(self, x):
        for i in range(self.n_layers):
            x = self.dense_layers[i](x)
        return x


class TransitionBlock(nn.Module):
    def __init__(self, growth_rate, n_dense_layers):
        super(TransitionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=growth_rate+(growth_rate * n_dense_layers * 2), out_channels=growth_rate, kernel_size=1, stride=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(x)
        out = self.avg_pool(out)
        return out


class DenseNet(nn.Module):

    def __init__(self, hp):
        super(DenseNet, self).__init__()
        self.hp = hp
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=7, stride=2, padding=4)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.dense_blocks = nn.ModuleList()
        self.transition_blocks = nn.ModuleList()

        for i in range(self.hp['num_dense_blocks']):
            self.dense_blocks.append(DenseBlock(growth_rate=self.hp['growth_rate'], n_layers=self.hp['num_dense_layers'][i]))
            if i < self.hp['num_dense_blocks'] - 1:
                self.transition_blocks.append((TransitionBlock(self.hp['growth_rate'], self.hp['num_dense_layers'][i])))

        self.fc1 = nn.Linear(in_features=84, out_features=10)

    def forward(self, img):
        out = self.conv1(img)
        out = self.maxpool1(out)
        for i in range(self.hp['num_dense_blocks']):
            out = self.dense_blocks[i](out)
            if i < self.hp['num_dense_blocks'] - 1:
                out = self.transition_blocks[i](out)
        out = torch.mean(out, dim=(2, 3))
        out = torch.reshape(out, [out.size(0), -1])
        logits = self.fc1(out)
        return logits


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, prediction, target):
        loss = self.ce_loss(prediction, target)
        return loss