import torch.nn as nn
import MinkowskiEngine as ME  


class Encoder(nn.Module):

    CHANNELS = [16, 32, 64, 128, 256, 512, 1024]
    DIMENSION = 3
    def __init__(self):
        nn.Module.__init__(self)
        ch = self.CHANNELS
        dimension = self.DIMENSION
        # Block 1
        self.block1 = nn.Sequential(
            ME.MinkowskiConvolution(1, ch[0], kernel_size=3, stride=2, dimension=dimension),
            ME.MinkowskiBatchNorm(ch[0]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[0], ch[0], kernel_size=3, dimension=dimension),
            ME.MinkowskiBatchNorm(ch[0]),
            ME.MinkowskiELU(),
        )

        self.block2 = nn.Sequential(
            ME.MinkowskiConvolution(ch[0], ch[1], kernel_size=3, stride=2, dimension=dimension),
            ME.MinkowskiBatchNorm(ch[1]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[1], ch[1], kernel_size=3, dimension=dimension),
            ME.MinkowskiBatchNorm(ch[1]),
            ME.MinkowskiELU(),
        )

        self.block3 = nn.Sequential(
            ME.MinkowskiConvolution(ch[1], ch[2], kernel_size=3, stride=2, dimension=dimension),
            ME.MinkowskiBatchNorm(ch[2]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[2], ch[2], kernel_size=3, dimension=dimension),
            ME.MinkowskiBatchNorm(ch[2]),
            ME.MinkowskiELU(),
        )

        self.block4 = nn.Sequential(
            ME.MinkowskiConvolution(ch[2], ch[3], kernel_size=3, stride=2, dimension=dimension),
            ME.MinkowskiBatchNorm(ch[3]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[3], ch[3], kernel_size=3, dimension=dimension),
            ME.MinkowskiBatchNorm(ch[3]),
            ME.MinkowskiELU(),
        )

        self.block5 = nn.Sequential(
            ME.MinkowskiConvolution(ch[3], ch[4], kernel_size=3, stride=2, dimension=dimension),
            ME.MinkowskiBatchNorm(ch[4]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[4], ch[4], kernel_size=3, dimension=dimension),
            ME.MinkowskiBatchNorm(ch[4]),
            ME.MinkowskiELU(),
        )

        self.block6 = nn.Sequential(
            ME.MinkowskiConvolution(ch[4], ch[5], kernel_size=3, stride=2, dimension=dimension),
            ME.MinkowskiBatchNorm(ch[5]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[5], ch[5], kernel_size=3, dimension=dimension),
            ME.MinkowskiBatchNorm(ch[5]),
            ME.MinkowskiELU(), 
            ME.MinkowskiConvolution(ch[5], ch[6], kernel_size=1, dimension=dimension)
        )
    

        self.block7 = nn.Sequential(
            ME.MinkowskiConvolution(ch[5], ch[6], kernel_size=3, stride=2, dimension=dimension),
            ME.MinkowskiBatchNorm(ch[6]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[6], ch[6], kernel_size=3, dimension=dimension),
            ME.MinkowskiBatchNorm(ch[6]),
            ME.MinkowskiELU(),
            
        )

        self.global_pool = ME.MinkowskiGlobalPooling()
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self, sinput):
        out = self.block1(sinput)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        block_6 = self.block6(out)
        out = self.global_pool(block_6)
        # out = self.block7(block_6)
        return out