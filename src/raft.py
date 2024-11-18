import torch
import torch.nn as nn

class RAFT(nn.Module):
    def __init__(self, device, args, corr_radius=3, corr_levels=4):
        super(RAFT, self).__init__()
        self.device = device
        self.corr_radius = corr_radius
        self.corr_levels = corr_levels
        self.iters = args.iters

        self.hdim= 96
        self.cdim = 64

        self.feature_encoder = FeatureEncoder(self.device, input_dim=3, output_dim=128).to(self.device)
        self.context_encoder = ContextEncoder(self.device, input_dim=3, output_dim=self.hdim+self.cdim).to(self.device)
        if 'attention' in args.name: 
            self.context_attention = ContextAttention(self.hdim)
        else:
            self.context_attention = None
        self.update_block = UpdateBlock(self.device, self.corr_radius, self.corr_levels, self.cdim, self.hdim, gru_output_dim=128, output_dim=2).to(self.device)

    def initialize_grids(self, image1, image2):
        N1, _, H1, W1 = image1.shape
        grid_x, grid_y = torch.meshgrid(torch.arange(H1//8), torch.arange(W1//8))
        coord_grid_1 = torch.stack((grid_x, grid_y), dim=0).unsqueeze(0)
        coord_grid_1 = coord_grid_1.repeat(N1, 1, 1, 1).float()

        N2, _, H2, W2 = image2.shape
        grid_x, grid_y = torch.meshgrid(torch.arange(H2//8), torch.arange(W2//8))
        coord_grid_2 = torch.stack((grid_x, grid_y), dim=0).unsqueeze(0)
        coord_grid_2 = coord_grid_2.repeat(N2, 1, 1, 1).float()

        return coord_grid_1, coord_grid_2

    def upsample_flow(self, flow, factor=8):
        _, _, H, W = flow.shape

        upsample_size = (H*factor, W*factor)
        upsampled_flow = nn.functional.interpolate(flow, size=upsample_size, mode='bilinear', align_corners=True)

        return (factor * upsampled_flow)

    def forward(self, image1, image2):
        
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        fmap1, fmap2 = self.feature_encoder.forward(image1, image2)

        cmap1 = self.context_encoder.forward(image1)
        cmap1_h, cmap1_c = torch.split(cmap1, [self.hdim, self.cdim], dim=1)
        cmap1_h = torch.tanh(cmap1_h)
        if self.context_attention is not None: 
            cmap1_h = self.context_attention(cmap1_h)
        cmap1_c = torch.relu(cmap1_c)

        coord_grid_1, coord_grid_2 = self.initialize_grids(image1, image2)
        coord_grid_1, coord_grid_2 = coord_grid_1.to(self.device), coord_grid_2.to(self.device)

        # Initialize Correlation Block
        corr_block = CorrelationBlock(self.device, fmap1, fmap2, self.corr_radius, self.corr_levels).to(self.device)

        flow_predictions = []
        for _ in range(self.iters):

            # Generate Correlation Volume
            coord_grid_2 = coord_grid_2.detach()
            corr = corr_block.compute_corr(coord_grid_2)

            flow = coord_grid_2 - coord_grid_1

            cmap1_h, del_flow = self.update_block.forward(cmap1_h, cmap1_c, corr, flow)

            coord_grid_2 = coord_grid_2 + del_flow
            updated_flow = coord_grid_2 - coord_grid_1

            upsampled_flow = self.upsample_flow(updated_flow, 8)

            flow_predictions.append(upsampled_flow)

        return flow_predictions


class BottleneckLayer(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', stride=1):
        super(BottleneckLayer, self).__init__()
        self.downsample = False

        self.conv1 = nn.Conv2d(input_dim, output_dim//4, (1,1), 1, 0)
        self.conv2 = nn.Conv2d(output_dim//4, output_dim//4, (3,3), stride, 1)
        self.conv3 = nn.Conv2d(output_dim//4, output_dim, (1,1), 1, 0)
        
        if norm == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            self.norm4 = nn.Sequential()
        elif norm == 'instance':
            self.norm1 = nn.InstanceNorm2d(output_dim//4)
            self.norm2 = nn.InstanceNorm2d(output_dim//4)
            self.norm3 = nn.InstanceNorm2d(output_dim)
            self.norm4 = nn.InstanceNorm2d(output_dim)

        self.relu = nn.ReLU(inplace=True)

        if stride != 1:
            self.downsample = True
            self.conv4 = nn.Conv2d(input_dim, output_dim, (1,1), stride, 0)
            

    def forward(self, x):
        y = self.relu(self.norm1(self.conv1(x)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))

        if self.downsample:
            z = self.norm4(self.conv4(x))
            y_out = self.relu(z+y)
        else:
            y_out = self.relu(x+y)

        return y_out


class FeatureEncoder(nn.Module):
    def __init__(self, device, input_dim=3, output_dim=128):
        super(FeatureEncoder, self).__init__()
        self.conv1 =  nn.Conv2d(input_dim, 32, (7, 7), 2, 3)
        self.norm = nn.InstanceNorm2d(32)
        self.relu = nn.ReLU()

        #ResUnits Layers
        self.res_unit1 = nn.Sequential(
            BottleneckLayer(32, 32, norm='instance', stride=1),
            BottleneckLayer(32, 32, norm='instance', stride=1))

        self.res_unit2 = nn.Sequential(
            BottleneckLayer(32, 64, norm='instance', stride=2),
            BottleneckLayer(64, 64, norm='instance', stride=1))

        self.res_unit3 = nn.Sequential(
            BottleneckLayer(64, 96, norm='instance', stride=2),
            BottleneckLayer(96, 96, norm='instance', stride=1))
        
        self.conv2 = nn.Conv2d(96, output_dim, (1,1), 1, 0)
    
    def forward(self, image1, image2):
        x = torch.cat((image1, image2), dim=0)

        x = self.relu(self.norm(self.conv1(x)))
        x = self.res_unit1(x)
        x = self.res_unit2(x)
        x = self.res_unit3(x)
        x = self.conv2(x)

        fmap1, fmap2 = torch.split(x, [image1.shape[0], image2.shape[0]], dim=0)

        return fmap1, fmap2


class ContextAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.conv_query = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_key = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_value = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        N, C, H, W = x.shape

        q = self.conv_query(x).reshape(N, C, H*W)
        k = self.conv_key(x).reshape(N, C, H*W)
        v = self.conv_value(x).reshape(N, C, H*W)
        temp = self.softmax(torch.matmul(k.transpose(1,2), q)/ (C ** 0.5))
        attention = torch.matmul(v, temp)
        attention = attention.reshape(N, C, H, W)
        return x + attention

class ContextEncoder(nn.Module):
    def __init__(self, device, input_dim=3, output_dim=128):
        super(ContextEncoder, self).__init__()
        self.conv1 =  nn.Conv2d(input_dim, 32, (7, 7), 2, 3)
        self.relu = nn.ReLU()

        #ResUnits Layers
        self.res_unit1 = nn.Sequential(
            BottleneckLayer(32, 32, norm='none', stride=1),
            BottleneckLayer(32, 32, norm='none', stride=1))

        self.res_unit2 = nn.Sequential(
            BottleneckLayer(32, 64, norm='none', stride=2),
            BottleneckLayer(64, 64, norm='none', stride=1))

        self.res_unit3 = nn.Sequential(
            BottleneckLayer(64, 96, norm='none', stride=2),
            BottleneckLayer(96, 96, norm='none', stride=1))
        
        self.conv2 = nn.Conv2d(96, output_dim, (1,1), 1, 0)
    
    def forward(self, image1):
        y = self.relu(self.conv1(image1))
        y = self.res_unit1(y)
        y = self.res_unit2(y)
        y = self.res_unit3(y)
        cmap = self.conv2(y)

        return cmap


class CorrelationBlock(nn.Module):
    def __init__(self, device, fmap1, fmap2, corr_radius=3, corr_levels=4):
        super(CorrelationBlock, self).__init__()
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.corr_pyramid = []

        # all pairs correlation
        N1, C1, H1, W1 = fmap1.shape
        N2, C2, H2, W2 = fmap2.shape
        fmap1 = torch.reshape(fmap1, (N1, C1, -1)).transpose(1,2)
        fmap2 = torch.reshape(fmap2, (N2, C2, -1))
        
        corr = torch.matmul(fmap1, fmap2)
        corr = torch.reshape(corr, (N1, H1, W1, 1, H2, W2))
        corr = corr / float(C1 ** 0.5)
        corr = torch.reshape(corr, (N1*H1*W1, 1, H2, W2))
        
        avg_pool2d = nn.AvgPool2d(2, 2)

        self.corr_pyramid.append(corr)
        for _ in range(self.corr_levels - 1):
            corr = avg_pool2d(corr)
            self.corr_pyramid.append(corr)

    def compute_corr(self, coord_grid):
        coord_grid = coord_grid.transpose(1, 2).transpose(2, 3)
        N, H, W, _ = coord_grid.shape

        pyramid = []
        for i in range(self.corr_levels):
            corr = self.corr_pyramid[i]
            dx = torch.arange(0, (2*self.corr_radius)+1)
            dy = torch.arange(0, (2*self.corr_radius)+1)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coord_grid.device)

            sub_coord_grid = (torch.reshape(coord_grid, (N*H*W, 1, 1, 2)) / 2**i) + torch.reshape(delta, (1, (2*self.corr_radius)+1, (2*self.corr_radius)+1, 2))

            corr = self.bilinear_sampler(corr, sub_coord_grid)
            corr = torch.reshape(corr, (N, H, W, -1))
            pyramid.append(corr)

        correlation = torch.cat(pyramid, dim=-1).transpose(1, 2).transpose(1, 3).float()

        return correlation
        
    def bilinear_sampler(self, corr, sub_coord_grid, mode='bilinear', mask=False):
        H, W = corr.shape[-2:]
        xgrid, ygrid = torch.split(sub_coord_grid, split_size_or_sections=1, dim=-1)
        xgrid = 2*xgrid/(W-1) - 1
        ygrid = 2*ygrid/(H-1) - 1

        grid = torch.cat([xgrid, ygrid], dim=-1)
        corr = nn.functional.grid_sample(corr, grid, align_corners=True)

        return corr

class UpdateBlock(nn.Module):
    def __init__(self, device, corr_radius=3, corr_levels=4, cdim=64, hdim=96, gru_output_dim=128, output_dim=2):
        super(UpdateBlock, self).__init__()
        cor_planes = corr_levels * (2*corr_radius + 1)**2
        
        self.relu = nn.ReLU()

        # Motion Encoder
        self.corr = nn.Sequential(
            nn.Conv2d(cor_planes, 96, (1,1), 1, 0),
            nn.ReLU()
        )

        self.flow = nn.Sequential(
            nn.Conv2d(2, 64, (7,7), 1, 3),
            nn.ReLU(),
            nn.Conv2d(64, 32, (3,3), 1, 1),
            nn.ReLU()
        )

        self.conv = nn.Conv2d(128, 80, (3,3), 1, 1)

        # ConvGRU layers
        motion_features_dim = 82
        self.convz = nn.Conv2d(hdim+cdim+motion_features_dim, hdim, (3,3), 1, 1)
        self.convr = nn.Conv2d(hdim+cdim+motion_features_dim, hdim, (3,3), 1, 1)
        self.convq = nn.Conv2d(hdim+cdim+motion_features_dim, hdim, (3,3), 1, 1)

        # Delta Flow
        self.conv1 = nn.Conv2d(hdim, 128, (3,3), 1, 1)
        self.conv2 = nn.Conv2d(128, 2, (3,3), 1, 1) # kernel size in code is 3

    def forward(self, cmap1_h, cmap1_c, corr, flow):

        # Compute Motion Features
        conv_corr = self.corr(corr)
        conv_flow = self.flow(flow)
        corr_flow = torch.cat([conv_corr, conv_flow], dim=1)
        conv_corr_flow = self.relu(self.conv(corr_flow))
 
        # Concatenate conv_corr_flow and unconvolved flow
        motion_features = torch.cat([conv_corr_flow, flow], dim=1)

        # Concatenate context dimension and motion features to input into GRU block
        cmap1_c = torch.cat([cmap1_c, motion_features], dim=1)

        # Apply ConvGRU on the context hdim and updated cdim to get updated hdim
        # [H_t-1, X_t]
        context = torch.cat([cmap1_h, cmap1_c], dim=1)
        #Z_t = sigmoid(Conv(W_z, [H_t-1, X_t]))
        z = torch.sigmoid(self.convz(context))
        #R_t = sigmoid(Conv(W_r, [H_t-1, X_t]))
        r = torch.sigmoid(self.convr(context))
        #Q~_t = tanh(Conv(W_h, [R_t * H_t-1, X_t]))
        q = torch.tanh(self.convq(torch.cat([r*cmap1_h, cmap1_c], dim=1)))
        #H_t = (1 - Z_t) * H_t-1 + Z_t * H~_t
        cmap1_h = (1-z) * cmap1_h + z * q

        # Calculate delta flow
        del_flow = self.conv2(self.relu(self.conv1((cmap1_h))))

        return cmap1_h, del_flow
