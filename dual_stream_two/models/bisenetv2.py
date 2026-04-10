"""
Paper:      BiSeNet V2: Bilateral Network with Guided Aggregation for 
            Real-time Semantic Segmentation
Url:        https://arxiv.org/abs/2004.02147
Create by:  zh320
Date:       2023/04/15
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cdht.dht import DHT_Layer

from .modules import conv3x3, conv1x1, DWConvBNAct, PWConvBNAct, ConvBNAct, Activation, SegHead, DWDownsample, LearnedAnisoDownsample
from .model_registry import register_model, aux_models


class FuseReduce(nn.Module):
    """Concatenate two feature maps and reduce back to c_out."""
    def __init__(self, c_in, c_out, act_type='relu'):
        super().__init__()
        self.conv = ConvBNAct(c_in, c_out, 3, act_type=act_type)

    def forward(self, a, b):
        return self.conv(torch.cat([a, b], dim=1))


@register_model(aux_models)
class BiSeNetv2(nn.Module):
    def __init__(self, num_class=1, n_channel=3, act_type='relu', use_aux=True):
        super().__init__()
        self.use_aux = use_aux
        self.detail_branch = DetailBranch(n_channel, 128, act_type)
        self.semantic_branch = SemanticBranch(n_channel, 128, num_class, act_type, use_aux)
        self.bga_layer = BilateralGuidedAggregationLayer(128, 128, act_type)
        self.seg_head = SegHead(128, num_class, act_type)

    def forward(self, x, is_training=False):
        size = x.size()[2:]
        x_d = self.detail_branch(x)
        if self.use_aux:
            x_s, aux2, aux3, aux4, aux5 = self.semantic_branch(x)
        else:
            x_s = self.semantic_branch(x)
        x = self.bga_layer(x_d, x_s)
        x = self.seg_head(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        if self.use_aux and is_training:
            return x, (aux2, aux3, aux4, aux5)
        else:
            return x



@register_model(aux_models)
class BiSeNetv2Dual(nn.Module):
    """
    BiSeNetv2 with two input streams for processing pairs of images.
    Features from both streams are fused before the final segmentation head.
    """
    def __init__(self, num_class=1, n_channel=3, act_type='relu', use_aux=True):
        super().__init__()
        self.use_aux = use_aux
        
        # Stream 1
        print('setting up stream 1')
        self.detail_branch1 = DetailBranch(n_channel, 128, act_type)
        self.semantic_branch1 = SemanticBranch(n_channel, 128, num_class, act_type, use_aux)
        self.bga_layer1 = BilateralGuidedAggregationLayer(128, 128, act_type)
        
        # Stream 2
        print('setting up stream 2')
        self.detail_branch2 = DetailBranch(n_channel, 128, act_type)
        self.semantic_branch2 = SemanticBranch(n_channel, 128, num_class, act_type, use_aux)
        self.bga_layer2 = BilateralGuidedAggregationLayer(128, 128, act_type)

        # Fusion: Concatenate 128 + 128 -> 256, reduce to 128
        self.fusion_conv = ConvBNAct(256, 128, 3, act_type=act_type)
        self.seg_head = SegHead(128, num_class, act_type)

    def forward(self, x1, x2, is_training=False):
        size = x1.size()[2:]
        
        # Stream 1
        x_d1 = self.detail_branch1(x1)
        if self.use_aux:
            x_s1, aux2_1, aux3_1, aux4_1, aux5_1 = self.semantic_branch1(x1)
        else:
            x_s1 = self.semantic_branch1(x1)

        x1_feat = self.bga_layer1(x_d1, x_s1)        
        

        # Stream 2
        x_d2 = self.detail_branch2(x2)
        if self.use_aux:
            x_s2, aux2_2, aux3_2, aux4_2, aux5_2 = self.semantic_branch2(x2)
        else:
            x_s2 = self.semantic_branch2(x2)

        x2_feat = self.bga_layer2(x_d2, x_s2)

        # Fusion
        x = torch.cat([x1_feat, x2_feat], dim=1)
        x = self.fusion_conv(x)
        x = self.seg_head(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        if self.use_aux and is_training:
            # Return aux outputs from both streams
            return x, (aux2_1, aux3_1, aux4_1, aux5_1, aux2_2, aux3_2, aux4_2, aux5_2)
        else:
            return x
        

@register_model(aux_models)
class BiSeNetv2DualHT(nn.Module):
    """
    BiSeNetv2 with two input streams for processing pairs of images.
    Features from both streams are fused before the final segmentation head.
    """
    def __init__(self, num_class=1, n_channel=3, act_type='relu', use_aux=True):
        super().__init__()
        self.use_aux = use_aux
        
        # Stream 1
        print('setting up stream 1')
        self.detail_branch1 = DetailBranch(n_channel, 128, act_type)

        self.semantic_branch1 = SemanticBranch(n_channel, 128, num_class, act_type, use_aux)
        self.bga_layer1 = BilateralGuidedAggregationLayer(128, 128, act_type)
        
        # Stream 2
        print('setting up stream 2')
        self.detail_branch2 = DetailBranch_HT(n_channel, 128, act_type)
        self.semantic_branch2 = SemanticBranch_HT(n_channel, 128, num_class, act_type, use_aux)
        self.bga_layer2 = BilateralGuidedAggregationLayer(128, 128, act_type)

        # Multi-scale detail fusion (fused feature replaces stream1 at each scale)
        self.fuse_d2 = FuseReduce(64 * 2, 64, act_type)
        self.fuse_d4 = FuseReduce(64 * 2, 64, act_type)
        self.fuse_d8 = FuseReduce(128 * 2, 128, act_type)

        self.dht_layer_1 = DHT_Layer(64, 64, 180, 208)
        self.dht_layer_2 = DHT_Layer(64, 64, 180, 104)
        self.dht_layer_3 = DHT_Layer(128, 128, 180, 72)
        
        # Fusion: Concatenate 128 + 128 -> 256, reduce to 128
        # self.fusion_conv = ConvBNAct(256, 128, 3, act_type=act_type)
        self.seg_head = SegHead(128, num_class, act_type)

    def forward(self, x1, x2, is_training=False):
        size = x1.size()[2:]
        
        # Detail branches with multi-scale fusion (stream1 replaced by fused features)
        d1_layers = list(self.detail_branch1.children())
        d2_layers = list(self.detail_branch2.children())

        # /2
        x1_d = d1_layers[0](x1)
        x2_d = d2_layers[0](x2)
        # extract dht features here
        x2_d = self.dht_layer_1(x2_d)
        # then downsample with interpolation to match x1_d size
        x2_d = F.interpolate(x2_d, size=x1_d.shape[2:], mode='bilinear', align_corners=True)
        x1_d = self.fuse_d2(x1_d, x2_d)

        x1_d = d1_layers[1](x1_d)
        x2_d = d2_layers[1](x2_d)

        # /4
        x1_d = d1_layers[2](x1_d)
        x2_d = d2_layers[2](x2_d)
    # extract dht features here
        x2_d = self.dht_layer_2(x2_d)
        # then downsample with interpolation to match x1_d size
        x2_d = F.interpolate(x2_d, size=x1_d.shape[2:], mode='bilinear', align_corners=True)
        x1_d = self.fuse_d4(x1_d, x2_d)

        x1_d = d1_layers[3](x1_d)
        x2_d = d2_layers[3](x2_d)

        x1_d = d1_layers[4](x1_d)
        x2_d = d2_layers[4](x2_d)

        # /8
        x1_d = d1_layers[5](x1_d)
        x2_d = d2_layers[5](x2_d)
        # extract dht features here
        x2_d = self.dht_layer_3(x2_d)
        # then downsample with interpolation to match x1_d size
        x2_d = F.interpolate(x2_d, size=x1_d.shape[2:], mode='bilinear', align_corners=True)
        x1_d = self.fuse_d8(x1_d, x2_d)

        x1_d = d1_layers[6](x1_d)
        # x2_d = d2_layers[6](x2_d)
        x1_d = d1_layers[7](x1_d)
        # x2_d = d2_layers[7](x2_d)
        if len(d2_layers) > 8:
            x2_d = d2_layers[8](x2_d)

        # Semantic branches
        if self.use_aux:
            x_s1, aux2_1, aux3_1, aux4_1, aux5_1 = self.semantic_branch1(x1)
            x_s2, aux2_2, aux3_2, aux4_2, aux5_2 = self.semantic_branch2(x2)
        else:
            x_s1 = self.semantic_branch1(x1)
            # x_s2 = self.semantic_branch2(x2)

        x1_feat = self.bga_layer1(x1_d, x_s1)
        # x2_feat = self.bga_layer2(x2_d, x_s2)
        
        # Fusion
        # x = torch.cat([x1_feat, x2_feat], dim=1)
        # x = self.fusion_conv(x)
        x = self.seg_head(x1_feat)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        if self.use_aux and is_training:
            # Return aux outputs from both streams
            return x, (aux2_1, aux3_1, aux4_1, aux5_1, aux2_2, aux3_2, aux4_2, aux5_2)
        else:
            return x

@register_model(aux_models)
class BiSeNetv2DualHTlastlayer(nn.Module):
    """
    BiSeNetv2 with two input streams for processing pairs of images.
    Features from both streams are fused before the final segmentation head.
    """
    def __init__(self, num_class=1, n_channel=3, act_type='relu', use_aux=True):
        super().__init__()
        self.use_aux = use_aux
        
        # Stream 1
        print('setting up stream 1')
        self.detail_branch1 = DetailBranch(n_channel, 128, act_type)

        self.semantic_branch1 = SemanticBranch(n_channel, 128, num_class, act_type, use_aux)
        self.bga_layer1 = BilateralGuidedAggregationLayer(128, 128, act_type)
        
        # Stream 2
        print('setting up stream 2')
        self.detail_branch2 = DetailBranch_HT(n_channel, 128, act_type)
        self.semantic_branch2 = SemanticBranch_HT(n_channel, 128, num_class, act_type, use_aux)
        self.bga_layer2 = BilateralGuidedAggregationLayer(128, 128, act_type)

        # Multi-scale detail fusion (fused feature replaces stream1 at each scale)
        # self.fuse_d2 = FuseReduce(64 * 2, 64, act_type)
        # self.fuse_d4 = FuseReduce(64 * 2, 64, act_type)
        self.fuse_d8 = FuseReduce(128 * 2, 128, act_type)

        # self.dht_layer_1 = DHT_Layer(64, 64, 100, 200)
        # self.dht_layer_2 = DHT_Layer(64, 64, 100, 100)
        self.dht_layer = DHT_Layer(128, 128, 100, 72)
        
        # Fusion: Concatenate 128 + 128 -> 256, reduce to 128
        # self.fusion_conv = ConvBNAct(256, 128, 3, act_type=act_type)
        self.seg_head = SegHead(128, num_class, act_type)

    def forward(self, x1, x2, is_training=False):
        size = x1.size()[2:]
        
        # Detail branches with multi-scale fusion (stream1 replaced by fused features)
        d1_layers = list(self.detail_branch1.children())
        d2_layers = list(self.detail_branch2.children())

        # /2
        x1_d = d1_layers[0](x1)
        x2_d = d2_layers[0](x2)
        
        x1_d = d1_layers[1](x1_d)
        x2_d = d2_layers[1](x2_d)

        # /4
        x1_d = d1_layers[2](x1_d)
        x2_d = d2_layers[2](x2_d)

        x1_d = d1_layers[3](x1_d)
        x2_d = d2_layers[3](x2_d)

        x1_d = d1_layers[4](x1_d)
        x2_d = d2_layers[4](x2_d)

        # /8
        x1_d = d1_layers[5](x1_d)
        x2_d = d2_layers[5](x2_d)
        # extract dht features here
        # x2_d = self.dht_layer_3(x2_d)
        # then downsample with interpolation to match x1_d size
        # x2_d = F.interpolate(x2_d, size=x1_d.shape[2:], mode='bilinear', align_corners=True)
        # x1_d = self.fuse_d8(x1_d, x2_d)

        x1_d = d1_layers[6](x1_d)
        x2_d = d2_layers[6](x2_d)
        x1_d = d1_layers[7](x1_d)
        x2_d = d2_layers[7](x2_d)
        # x2_d = self.dht_layer_3(x2_d)
        # x2_d = F.interpolate(x2_d, size=x1_d.shape[2:], mode='bilinear', align_corners=True)
        # x1_d = self.fuse_d8(x1_d, x2_d)

        if len(d2_layers) > 8:
            x2_d = d2_layers[8](x2_d)

        # Semantic branches
        if self.use_aux:
            x_s1, aux2_1, aux3_1, aux4_1, aux5_1 = self.semantic_branch1(x1)
            x_s2, aux2_2, aux3_2, aux4_2, aux5_2 = self.semantic_branch2(x2)
        else:
            x_s1 = self.semantic_branch1(x1)
            x_s2 = self.semantic_branch2(x2)

        x1_feat = self.bga_layer1(x1_d, x_s1)
        x2_feat = self.bga_layer2(x2_d, x_s2)
        # extrac dht features here
        x2_feat = self.dht_layer(x2_feat)
        x2_feat = F.interpolate(x2_feat, size=x1_feat.shape[2:], mode='bilinear', align_corners=True)
        x1_feat = self.fuse_d8(x1_feat, x2_feat)
        
        # Fusion
        # x = torch.cat([x1_feat, x2_feat], dim=1)
        # x = self.fusion_conv(x)
        x = self.seg_head(x1_feat)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        if self.use_aux and is_training:
            # Return aux outputs from both streams
            return x, (aux2_1, aux3_1, aux4_1, aux5_1, aux2_2, aux3_2, aux4_2, aux5_2)
        else:
            return x

class DetailBranch(nn.Sequential):
    def __init__(self, in_channels, out_channels, act_type='relu'):
        super().__init__(
            ConvBNAct(in_channels, 64, 3, 2, act_type=act_type),
            ConvBNAct(64, 64, 3, 1, act_type=act_type),
            ConvBNAct(64, 64, 3, 2, act_type=act_type),
            ConvBNAct(64, 64, 3, 1, act_type=act_type),
            ConvBNAct(64, 128, 3, 1, act_type=act_type),
            ConvBNAct(128, 128, 3, 2, act_type=act_type),
            ConvBNAct(128, 128, 3, 1, act_type=act_type),
            ConvBNAct(128, out_channels, 3, 1, act_type=act_type)
        )

class DetailBranch_HT(nn.Sequential):
    def __init__(self, in_channels, out_channels, act_type='relu'):
        super().__init__(
            ConvBNAct(in_channels, 64, 3, 2, act_type=act_type),
            ConvBNAct(64, 64, 3, 1, act_type=act_type),
            ConvBNAct(64, 64, 3, 2, act_type=act_type),
            ConvBNAct(64, 64, 3, 1, act_type=act_type),
            ConvBNAct(64, 128, 3, 1, act_type=act_type),
            ConvBNAct(128, 128, 3, 2, act_type=act_type),
            ConvBNAct(128, 128, 3, 1, act_type=act_type),
            ConvBNAct(128, out_channels, 3, 1, act_type=act_type)
        )


class SemanticBranch_HT(nn.Sequential):
    def __init__(self, in_channels, out_channels, num_class, act_type='relu', use_aux=False):
        super().__init__()
        self.use_aux = use_aux
        self.stage1to2 = StemBlock(in_channels, 16, act_type)
        self.stage3 = nn.Sequential(
                            GatherExpansionLayer(16, 32, 2, act_type),
                            GatherExpansionLayer(32, 32, 1, act_type),
                        )
        self.stage4 = nn.Sequential(
                            GatherExpansionLayer(32, 64, 2, act_type),
                            GatherExpansionLayer(64, 64, 1, act_type),
                        )
        self.stage5_1to4 = nn.Sequential(
                                GatherExpansionLayer(64, 128, 2, act_type),
                                GatherExpansionLayer(128, 128, 1, act_type),
                                GatherExpansionLayer(128, 128, 1, act_type),
                                GatherExpansionLayer(128, 128, 1, act_type),
                            )
        self.stage5_5 = ContextEmbeddingBlock(128, out_channels, act_type)

        if self.use_aux:
            self.seg_head2 = SegHead(16, num_class, act_type)
            self.seg_head3 = SegHead(32, num_class, act_type)
            self.seg_head4 = SegHead(64, num_class, act_type)
            self.seg_head5 = SegHead(128, num_class, act_type)

    def forward(self, x):
        x = self.stage1to2(x)
        if self.use_aux:
            aux2 = self.seg_head2(x)

        x = self.stage3(x)
        if self.use_aux:
            aux3 = self.seg_head3(x)

        x = self.stage4(x)
        if self.use_aux:
            aux4 = self.seg_head4(x)

        x = self.stage5_1to4(x)
        if self.use_aux:
            aux5 = self.seg_head5(x)

        x = self.stage5_5(x)

        if self.use_aux:
            return x, aux2, aux3, aux4, aux5
        else:
            return x
        

class SemanticBranch(nn.Sequential):
    def __init__(self, in_channels, out_channels, num_class, act_type='relu', use_aux=False):
        super().__init__()
        self.use_aux = use_aux
        self.stage1to2 = StemBlock(in_channels, 16, act_type)
        self.stage3 = nn.Sequential(
                            GatherExpansionLayer(16, 32, 2, act_type),
                            GatherExpansionLayer(32, 32, 1, act_type),
                        )
        self.stage4 = nn.Sequential(
                            GatherExpansionLayer(32, 64, 2, act_type),
                            GatherExpansionLayer(64, 64, 1, act_type),
                        )
        self.stage5_1to4 = nn.Sequential(
                                GatherExpansionLayer(64, 128, 2, act_type),
                                GatherExpansionLayer(128, 128, 1, act_type),
                                GatherExpansionLayer(128, 128, 1, act_type),
                                GatherExpansionLayer(128, 128, 1, act_type),
                            )
        self.stage5_5 = ContextEmbeddingBlock(128, out_channels, act_type)

        if self.use_aux:
            self.seg_head2 = SegHead(16, num_class, act_type)
            self.seg_head3 = SegHead(32, num_class, act_type)
            self.seg_head4 = SegHead(64, num_class, act_type)
            self.seg_head5 = SegHead(128, num_class, act_type)

    def forward(self, x):
        x = self.stage1to2(x)
        if self.use_aux:
            aux2 = self.seg_head2(x)

        x = self.stage3(x)
        if self.use_aux:
            aux3 = self.seg_head3(x)

        x = self.stage4(x)
        if self.use_aux:
            aux4 = self.seg_head4(x)

        x = self.stage5_1to4(x)
        if self.use_aux:
            aux5 = self.seg_head5(x)

        x = self.stage5_5(x)

        if self.use_aux:
            return x, aux2, aux3, aux4, aux5
        else:
            return x


@register_model(aux_models)
class BiSeNetv2DualMaskGuidedv2(nn.Module):
    """
    Dual-stream variant where stream2 is supervised separately and its mask
    prediction guides stream1 via fusion before the final seg head.

    Differences vs BiSeNetv2Dual:
    - No feature fusion between streams prior to seg head; only stream1 features
      are fed to the final head.
    - Stream2 has its own segmentation head (and supervision).
    - Stream2's predicted mask is fused with stream1 features before stream1's
      seg head to guide the final prediction.
    """

    def __init__(self, num_class=1, n_channel=3, act_type='relu', use_aux=True):
        super().__init__()
        self.use_aux = use_aux

        # Stream 1
        self.detail_branch1 = DetailBranch(n_channel, 128, act_type)
        self.semantic_branch1 = SemanticBranch(n_channel, 128, num_class, act_type, use_aux)
        self.bga_layer1 = BilateralGuidedAggregationLayer(128, 128, act_type)

        # Stream 2 (has its own supervision)
        self.detail_branch2 = DetailBranch(n_channel, 128, act_type)
        self.semantic_branch2 = SemanticBranch(n_channel, 128, num_class, act_type, use_aux)
        self.bga_layer2 = BilateralGuidedAggregationLayer(128, 128, act_type)

        # Seg heads
        self.seg_head1 = SegHead(128, num_class, act_type)  # main output
        self.seg_head2 = SegHead(128, num_class, act_type)  # stream2 supervision

        # Optional downsampling of stream2 logits before fusion (supports anisotropic stride)
        # self.stream2_down = DWDownsample(num_class, num_class, kernel_size=3, stride=(2, 5), act_type=act_type)
        self.stream2_down = LearnedAnisoDownsample(num_class, num_class)

        # Fuse stream1 features with stream2 mask logits before seg_head1
        # self.mask_fusion_l1 = ConvBNAct(2 * num_class, 8, 3, act_type=act_type)
        # self.mask_fusion_l2 = ConvBNAct(8, 16, 3, act_type=act_type)
        # self.mask_fusion_l3 = ConvBNAct(16, 32, 3, act_type=act_type)
        self.final_seghead = SegHead(2 * num_class, num_class, act_type, hid_channels=16)
        # add a couple more layers here

        self.dht_layer = DHT_Layer(num_class, num_class, 192, 288)
        # self.dht_layer = DHT_Layer(num_class, num_class, 192, 416)

    def forward(self, x1, x2, is_training=False):
        size_1 = x1.size()[2:]
        size_2 = x2.size()[2:]

        # Stream 1
        x_d1 = self.detail_branch1(x1)
        if self.use_aux:
            x_s1, aux2_1, aux3_1, aux4_1, aux5_1 = self.semantic_branch1(x1)
        else:
            x_s1 = self.semantic_branch1(x1)
        x1_feat = self.bga_layer1(x_d1, x_s1)

        # Stream 2 (supervised)
        x_d2 = self.detail_branch2(x2)
        if self.use_aux:
            x_s2, aux2_2, aux3_2, aux4_2, aux5_2 = self.semantic_branch2(x2)
        else:
            x_s2 = self.semantic_branch2(x2)
        x2_feat = self.bga_layer2(x_d2, x_s2)

        # Stream2 mask prediction (logits at feature resolution)
        stream2_logits = self.seg_head2(x2_feat)

        main_logits = self.seg_head1(x1_feat)

        # Upsample both outputs to input resolution
        stream2_out = F.interpolate(stream2_logits, size_2, mode='bilinear', align_corners=True)
        stream2_to_be_fused = self.dht_layer(stream2_out)


        main_out = F.interpolate(main_logits, size_1, mode='bilinear', align_corners=True)
        fused = torch.cat([main_out, stream2_to_be_fused], dim=1)
        # main_out = self.mask_fusion_l1(fused)
        # main_out = self.mask_fusion_l2(main_out)
        # main_out = self.mask_fusion_l3(main_out)
        main_out = self.final_seghead(fused)

        if self.use_aux and is_training:
            return main_out, stream2_out, (aux2_1, aux3_1, aux4_1, aux5_1, aux2_2, aux3_2, aux4_2, aux5_2)
        else:
            return main_out, stream2_out
        

@register_model(aux_models)
class BiSeNetv2DualMaskGuided(nn.Module):
    """
    Dual-stream variant where stream2 is supervised separately and its mask
    prediction guides stream1 via fusion before the final seg head.

    Differences vs BiSeNetv2Dual:
    - No feature fusion between streams prior to seg head; only stream1 features
      are fed to the final head.
    - Stream2 has its own segmentation head (and supervision).
    - Stream2's predicted mask is fused with stream1 features before stream1's
      seg head to guide the final prediction.
    """

    def __init__(self, num_class=1, n_channel=3, act_type='relu', use_aux=True):
        super().__init__()
        self.use_aux = use_aux

        # Stream 1
        self.detail_branch1 = DetailBranch(n_channel, 128, act_type)
        self.semantic_branch1 = SemanticBranch(n_channel, 128, num_class, act_type, use_aux)
        self.bga_layer1 = BilateralGuidedAggregationLayer(128, 128, act_type)

        # Stream 2 (has its own supervision)
        self.detail_branch2 = DetailBranch(n_channel, 128, act_type)
        self.semantic_branch2 = SemanticBranch(n_channel, 128, num_class, act_type, use_aux)
        self.bga_layer2 = BilateralGuidedAggregationLayer(128, 128, act_type)

        # Seg heads
        self.seg_head1 = SegHead(128, num_class, act_type)  # main output
        self.seg_head2 = SegHead(128, num_class, act_type)  # stream2 supervision

        # Optional downsampling of stream2 logits before fusion (supports anisotropic stride)
        # self.stream2_down = DWDownsample(num_class, num_class, kernel_size=3, stride=(2, 5), act_type=act_type)
        self.stream2_down = LearnedAnisoDownsample(num_class, num_class)

        self.dht_layer = DHT_Layer(num_class, num_class, 192, 416)

        self.mask_fusion = ConvBNAct(2 * num_class, 2, 3, act_type=act_type)

    def forward(self, x1, x2, is_training=False):
        size_1 = x1.size()[2:]
        size_2 = x2.size()[2:]

        # Stream 1
        x_d1 = self.detail_branch1(x1)
        if self.use_aux:
            x_s1, aux2_1, aux3_1, aux4_1, aux5_1 = self.semantic_branch1(x1)
        else:
            x_s1 = self.semantic_branch1(x1)
        x1_feat = self.bga_layer1(x_d1, x_s1)

        # Stream 2 (supervised)
        x_d2 = self.detail_branch2(x2)
        if self.use_aux:
            x_s2, aux2_2, aux3_2, aux4_2, aux5_2 = self.semantic_branch2(x2)
        else:
            x_s2 = self.semantic_branch2(x2)
        x2_feat = self.bga_layer2(x_d2, x_s2)

        # Stream2 mask prediction (logits at feature resolution)
        stream2_logits = self.seg_head2(x2_feat)

        main_logits = self.seg_head1(x1_feat)

        # Upsample both outputs to input resolution
        stream2_out = F.interpolate(stream2_logits, size_2, mode='bilinear', align_corners=True)
        stream2_to_be_fused = self.dht_layer(stream2_out)


        main_out = F.interpolate(main_logits, size_1, mode='bilinear', align_corners=True)
        fused = torch.cat([main_out, stream2_to_be_fused], dim=1)
        main_out = self.mask_fusion(fused)
        

        if self.use_aux and is_training:
            return main_out, stream2_out, (aux2_1, aux3_1, aux4_1, aux5_1, aux2_2, aux3_2, aux4_2, aux5_2)
        else:
            return main_out, stream2_out
        
@register_model(aux_models)
class BiSeNetv2DualMaskGuidedv1(nn.Module):
    """
    Dual-stream variant where stream2 is supervised separately and its mask
    prediction guides stream1 via fusion before the final seg head.

    Differences vs BiSeNetv2Dual:
    - No feature fusion between streams prior to seg head; only stream1 features
      are fed to the final head.
    - Stream2 has its own segmentation head (and supervision).
    - Stream2's predicted mask is fused with stream1 features before stream1's
      seg head to guide the final prediction.
    """

    def __init__(self, num_class=1, n_channel=3, act_type='relu', use_aux=True):
        super().__init__()
        self.use_aux = use_aux

        # Stream 1
        self.detail_branch1 = DetailBranch(n_channel, 128, act_type)
        self.semantic_branch1 = SemanticBranch(n_channel, 128, num_class, act_type, use_aux)
        self.bga_layer1 = BilateralGuidedAggregationLayer(128, 128, act_type)

        # Stream 2 (has its own supervision)
        self.detail_branch2 = DetailBranch(n_channel, 128, act_type)
        self.semantic_branch2 = SemanticBranch(n_channel, 128, num_class, act_type, use_aux)
        self.bga_layer2 = BilateralGuidedAggregationLayer(128, 128, act_type)

        # Seg heads
        self.seg_head1 = SegHead(128, num_class, act_type)  # main output
        self.seg_head2 = SegHead(128, num_class, act_type)  # stream2 supervision

        # Optional downsampling of stream2 logits before fusion (supports anisotropic stride)
        # self.stream2_down = DWDownsample(num_class, num_class, kernel_size=3, stride=(2, 5), act_type=act_type)
        self.stream2_down = LearnedAnisoDownsample(num_class, num_class)

        # Fuse stream1 features with stream2 mask logits before seg_head1
        self.mask_fusion = ConvBNAct(128 + num_class, 128, 3, act_type=act_type)

        self.dht_layer = DHT_Layer(num_class, num_class, 180, 52)

    def forward(self, x1, x2, is_training=False):
        size_1 = x1.size()[2:]
        size_2 = x2.size()[2:]

        # Stream 1
        x_d1 = self.detail_branch1(x1)
        if self.use_aux:
            x_s1, aux2_1, aux3_1, aux4_1, aux5_1 = self.semantic_branch1(x1)
        else:
            x_s1 = self.semantic_branch1(x1)
        x1_feat = self.bga_layer1(x_d1, x_s1)

        # Stream 2 (supervised)
        x_d2 = self.detail_branch2(x2)
        if self.use_aux:
            x_s2, aux2_2, aux3_2, aux4_2, aux5_2 = self.semantic_branch2(x2)
        else:
            x_s2 = self.semantic_branch2(x2)
        x2_feat = self.bga_layer2(x_d2, x_s2)

        # Stream2 mask prediction (logits at feature resolution)
        stream2_logits = self.seg_head2(x2_feat)
        stream2_logits = self.dht_layer(stream2_logits)
        stream2_logits = self.stream2_down(stream2_logits)

        # Downsample then resize to match x1_feat spatial dimensions if needed
        stream2_logits = self.stream2_down(stream2_logits)
        if stream2_logits.size()[2:] != x1_feat.size()[2:]:
            stream2_logits = F.interpolate(stream2_logits, size=x1_feat.size()[2:], mode='bilinear', align_corners=True)

        # Fuse stream1 features with stream2 logits, then predict main mask
        fused = torch.cat([x1_feat, stream2_logits], dim=1)
        fused = self.mask_fusion(fused)
        main_logits = self.seg_head1(fused)

        # Upsample both outputs to input resolution
        main_out = F.interpolate(main_logits, size_1, mode='bilinear', align_corners=True)
        stream2_out = F.interpolate(stream2_logits, size_2, mode='bilinear', align_corners=True)

        if self.use_aux and is_training:
            return main_out, stream2_out, (aux2_1, aux3_1, aux4_1, aux5_1, aux2_2, aux3_2, aux4_2, aux5_2)
        else:
            return main_out, stream2_out

class StemBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_type='relu'):
        super().__init__()
        self.conv_init = ConvBNAct(in_channels, out_channels, 3, 2, act_type=act_type)
        self.left_branch = nn.Sequential(
                            ConvBNAct(out_channels, out_channels//2, 1, act_type=act_type),
                            ConvBNAct(out_channels//2, out_channels, 3, 2, act_type=act_type)
                    )
        self.right_branch = nn.MaxPool2d(3, 2, 1)
        self.conv_last = ConvBNAct(out_channels*2, out_channels, 3, 1, act_type=act_type)

    def forward(self, x):
        x = self.conv_init(x)
        x_left = self.left_branch(x)
        x_right = self.right_branch(x)
        x = torch.cat([x_left, x_right], dim=1)
        x = self.conv_last(x)

        return x


class GatherExpansionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride, act_type='relu', expand_ratio=6,):
        super().__init__()
        self.stride = stride
        hid_channels = int(round(in_channels * expand_ratio))

        layers = [ConvBNAct(in_channels, in_channels, 3, act_type=act_type)]

        if stride == 2:
            layers.extend([
                            DWConvBNAct(in_channels, hid_channels, 3, 2, act_type='none'),
                            DWConvBNAct(hid_channels, hid_channels, 3, 1, act_type='none')
                        ])
            self.right_branch = nn.Sequential(
                                    DWConvBNAct(in_channels, in_channels, 3, 2, act_type='none'),
                                    PWConvBNAct(in_channels, out_channels, act_type='none')
                            )            
        else:
            layers.append(DWConvBNAct(in_channels, hid_channels, 3, 1, act_type='none'))

        layers.append(PWConvBNAct(hid_channels, out_channels, act_type='none'))
        self.left_branch = nn.Sequential(*layers)
        self.act = Activation(act_type)

    def forward(self, x):
        res = self.left_branch(x)

        if self.stride == 2:
            res = self.right_branch(x) + res
        else:
            res = x + res

        return self.act(res)


class ContextEmbeddingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_type='relu'):
        super().__init__()
        self.pool = nn.Sequential(
                            nn.AdaptiveAvgPool2d(1),
                            nn.BatchNorm2d(in_channels)
                    )
        self.conv_mid = ConvBNAct(in_channels, in_channels, 1, act_type=act_type)
        self.conv_last = conv3x3(in_channels, out_channels)

    def forward(self, x):
        res = self.pool(x)
        res = self.conv_mid(res)
        x = res + x
        x = self.conv_last(x)

        return x


class BilateralGuidedAggregationLayer(nn.Module):
    def __init__(self, in_channels, out_channels, act_type='relu'):
        super().__init__()
        self.detail_high = nn.Sequential(
                                    DWConvBNAct(in_channels, in_channels, 3, act_type=act_type),
                                    conv1x1(in_channels, in_channels)
                        )
        self.detail_low = nn.Sequential(
                                    DWConvBNAct(in_channels, in_channels, 3, 2, act_type=act_type),
                                    nn.AvgPool2d(3, 2, 1)
                        )
        self.semantic_high = nn.Sequential(
                                    ConvBNAct(in_channels, in_channels, 3, act_type=act_type),
                                    nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
                                    nn.Sigmoid()
                            )
        self.semantic_low = nn.Sequential(
                                    DWConvBNAct(in_channels, in_channels, 3, act_type=act_type),
                                    conv1x1(in_channels, in_channels),
                                    nn.Sigmoid()
                            )
        self.conv_last = ConvBNAct(in_channels, out_channels, 3, act_type=act_type)

    def forward(self, x_d, x_s):
        x_d_high = self.detail_high(x_d)
        x_d_low = self.detail_low(x_d)

        x_s_high = self.semantic_high(x_s)
        x_s_low = self.semantic_low(x_s)
        x_high = x_d_high * x_s_high
        x_low = x_d_low * x_s_low

        size = x_high.size()[2:]
        x_low = F.interpolate(x_low, size, mode='bilinear', align_corners=True)
        res = x_high + x_low
        res = self.conv_last(res)

        return res