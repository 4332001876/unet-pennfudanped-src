import torch
import torch.nn as nn
import torch.nn.functional as F

def double_convolution(in_channels, out_channels):
    """
    In the original paper implementation, the convolution operations were
    not padded but we are padding them here. This is because, we need the 
    output result size to be same as input size.
    """
    conv_op = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
    return conv_op


class UNet3(nn.Module):
    def __init__(self, num_classes):
        super(UNet3, self).__init__()

        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)

        # Contracting path.
        # Each convolution is applied twice.
        self.down_convolution_1 = double_convolution(3, 64)
        self.down_convolution_2 = double_convolution(64, 128)
        self.down_convolution_3 = double_convolution(128, 256)

        # Expanding path.
        self.up_transpose_3 = nn.ConvTranspose2d(
            in_channels=256, 
            out_channels=128,
            kernel_size=2, 
            stride=2)
        self.up_convolution_3 = double_convolution(256, 128)
        self.up_transpose_4 = nn.ConvTranspose2d(
            in_channels=128, 
            out_channels=64,
            kernel_size=2, 
            stride=2)
        self.up_convolution_4 = double_convolution(128, 64)

        # output => increase the `out_channels` as per the number of classes.
        self.out = nn.Conv2d(
            in_channels=64, 
            out_channels=num_classes, 
            kernel_size=1
        ) 

    def forward(self, x):
        down_1 = self.down_convolution_1(x)
        down_2 = self.max_pool2d(down_1) # 1/2
        down_3 = self.down_convolution_2(down_2)
        down_4 = self.max_pool2d(down_3) # 1/4
        down_5 = self.down_convolution_3(down_4)
        
        up_3 = self.up_transpose_3(down_5) # 1/2
        x = self.up_convolution_3(torch.cat([down_3, up_3], 1))

        up_4 = self.up_transpose_4(x) # 1/1
        x = self.up_convolution_4(torch.cat([down_1, up_4], 1))

        out = self.out(x)
        return out

if __name__ == '__main__':
    input_image = torch.rand((1, 3, 512, 512))
    model = UNet3(num_classes=10)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    outputs = model(input_image)
    print(outputs.shape)