
import torch
from torch import nn
import torch.nn.functional as F
from utils.tensorize_batch import tensorize_batch
from common_blocks.depth_wise_conv import depth_wise_conv
from common_blocks.depth_wise_sep_conv import depth_wise_sep_conv
from common_blocks.continuous_conv import ContinuousConvolution
from segmentation_heads.sem_seg import segmentation_head as sem_seg_head
from segmentation_heads.sem_seg_with_depth import segmentation_head as sem_seg_with_depth
from models_bank.maskrcnn.detection.backbone_utils import resnet_fpn_backbone


import config_kitti
import matplotlib.pyplot as plt

"""
Backbone --> Depth head
Backbone --> Semantig Seg head
Depth head --> Semantic head
"""

class Two_D_Branch(nn.Module):
    def __init__(self, backbone_out_channels):
        super(Two_D_Branch, self).__init__()

        self.conv1 = nn.Sequential(
            depth_wise_conv(backbone_out_channels, kernel_size=3, stride=2),
            nn.BatchNorm2d(backbone_out_channels)
        )

        self.conv2 = nn.Sequential(
            depth_wise_conv(backbone_out_channels, kernel_size=3, stride=1),
            nn.BatchNorm2d(backbone_out_channels)
        )

        self.conv3 = nn.Sequential(
            depth_wise_conv(backbone_out_channels, kernel_size=3, stride=1),
            nn.BatchNorm2d(backbone_out_channels)
        )

    def forward(self, features):

        original_shape = features.shape[2:]
        conv1_out = F.relu(self.conv1(features))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv2_out = F.interpolate(conv2_out, original_shape)
        conv3_out = F.relu(self.conv3(features))

        return conv2_out + conv3_out


class Three_D_Branch(nn.Module):
    def __init__(self, n_feat, k_number, n_number=None):
        super(Three_D_Branch, self).__init__()

        self.branch_3d_continuous = nn.Sequential(
            ContinuousConvolution(n_feat, k_number, n_number),
            ContinuousConvolution(n_feat, k_number, n_number)
        )

    def forward(self, feats, mask, coors, indices):
        """
        mask: B x H x W
        feats: B x C x H x W
        coors: B x N x 3 (points coordinates)
        indices: B x N x K (knn indices, aka. mask_knn)
        """

        B, C, _, _ = feats.shape
        feats_mask = feats.permute(0, 2, 3, 1)[mask].view(B, -1, C)
        br_3d, _, _ = self.branch_3d_continuous(
            (feats_mask, coors, indices))  # B x N x C
        br_3d = br_3d.view(-1, C)  # B*N x C

        out = torch.zeros_like(feats.permute(0, 2, 3, 1))  # B x H x W x C
        out[mask] = br_3d
        out = out.permute(0, 3, 1, 2)  # B x C x H x W

        return out


class FuseBlock(nn.Module):
    def __init__(self, nin, nout, k_number, n_number=None, extra_output_layer=False):
        super(FuseBlock, self).__init__()

        self.extra_output_layer = extra_output_layer
        self.branch_2d = Two_D_Branch(nin)

        self.branch_3d = Three_D_Branch(nin, k_number, n_number)

        self.output_layer = nn.Sequential(
            # depth_wise_conv(backbone_out_channels, kernel_size=3, stride=1, padding=1),
            # depth_wise_sep_conv(nin, nout, kernel_size=3, padding=1),
            nn.Conv2d(nin, nout, kernel_size=3, padding=1),
            nn.BatchNorm2d(nout)
        )

    def forward(self, *inputs):

        # mask: B x H x W
        # feats: B x C x H x W
        # coors: B x N x 3 (points coordinates)
        # indices: B x N x K (knn indices, aka. mask_knn)

        feats, mask, coors, k_nn_indices = inputs[0]
        y = self.branch_3d(feats, mask, coors, k_nn_indices) + \
            self.branch_2d(feats)

        y = F.relu(self.output_layer(y))

        if self.extra_output_layer:
            y = y + feats
            return (y, mask, coors, k_nn_indices)

        return (y, mask, coors, k_nn_indices)


class Semseg_Depth(nn.Module):
    def __init__(self, k_number,
                 backbone_out_channels,
                 num_ins_classes,
                 num_sem_classes,
                 original_image_size,
                 n_number=None):

        super(Semseg_Depth, self).__init__()

        # Depth completion ------------------

        # Depth head---------------------------------------------------------------
        
        self.depth_head = sem_seg_head(
        backbone_out_channels, 1, original_image_size, depthwise_conv=config_kitti.SEMANTIC_HEAD_DEPTHWISE_CONV)

        self.feat_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.sparse_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.rgbd_conv = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.fuse_conv = nn.Sequential(
            FuseBlock(64, 64, k_number, n_number=n_number),
            FuseBlock(64, 64, k_number, n_number=n_number,
                      extra_output_layer=True),
            FuseBlock(64, 64, k_number, n_number=n_number,
                      extra_output_layer=True),
            FuseBlock(64, 64, k_number, n_number=n_number,
                      extra_output_layer=True),
            FuseBlock(64, 64, k_number, n_number=n_number,
                      extra_output_layer=True),
            FuseBlock(64, 64, k_number, n_number=n_number,
                      extra_output_layer=True),
            FuseBlock(64, 64, k_number, n_number=n_number,
                      extra_output_layer=True),
            FuseBlock(64, 64, k_number, n_number=n_number,
                      extra_output_layer=True),
            FuseBlock(64, 64, k_number, n_number=n_number,
                      extra_output_layer=True),
            FuseBlock(64, 64, k_number, n_number=n_number,
                      extra_output_layer=True),
            FuseBlock(64, 64, k_number, n_number=n_number,
                      extra_output_layer=True),
            FuseBlock(64, 64, k_number, n_number=n_number,
                      extra_output_layer=True)
        )

        self.output_layer = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Semantic Segmentation 
        self.backbone = resnet_fpn_backbone('resnet50', False)
        # backbone.body.conv1 = nn.Conv2d(48, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)


        self.semantic_head = sem_seg_with_depth(
            backbone_out_channels,
            num_ins_classes + num_sem_classes + 1, original_image_size, depthwise_conv=config_kitti.SEMANTIC_HEAD_DEPTHWISE_CONV)

    def forward(self, img, sparse_depth, mask, coors, k_nn_indices, sparse_depth_gt=None, semantic_masks=None):
        """
        inputs:
        img: input rgb (B x 3 x H x W)
        sparse_depth: input sparse depth (B x 1 x H x W)
        coors: sparse 3D points (B x 3 x N)
        mask: mask_2d3d (B x H x W)
        indices: mask_knn (B x N x K)

        output:
        depth: completed depth
        """
        
        # Backbone

        backbone_feat = self.backbone(img)

        P4, P8, P16, P32 = backbone_feat['0'], backbone_feat['1'], backbone_feat['2'], backbone_feat['3']
        
        
  

        # Depth completion ------------------------
        _, H, W = mask.shape

        # depth head

        feat = self.depth_head(P4, P8, P16, P32)

        # feat conv

        y_feat = self.feat_conv(feat)


        # sparse depth branch
        y_sparse = self.sparse_conv(sparse_depth)  # B x 16 x H/2 x W/2

        # rgbd branch
        x_concat_d = torch.cat((img, sparse_depth), dim=1)
        y_rgbd = self.rgbd_conv(x_concat_d)  # B x 32 x H/2 x W/2

        y_rgbd_cat_y_sparse_cat_y_feat = torch.cat((y_rgbd, y_sparse, y_feat), dim=1)

        y_rgbd_cat_y_sparse_cat_y_feat = F.interpolate(y_rgbd_cat_y_sparse_cat_y_feat, (H, W))

        fused, _, _, _ = self.fuse_conv(
            (y_rgbd_cat_y_sparse_cat_y_feat, mask, coors, k_nn_indices))

        fused_out = self.output_layer(fused)

        #Semantic Segmentation
        semantic_logits = self.semantic_head(P4, P8, P16, P32, fused_out)


        # out = fused_out.squeeze_(1)
        out = torch.squeeze(fused_out, 1)
        if self.training:
            
            device = sparse_depth_gt.get_device()
            # Depth completion 
            mask_gt = torch.where(sparse_depth_gt > 0, torch.tensor((1), device=device,
                                                                    dtype=torch.float64), torch.tensor((0), device=device, dtype=torch.float64))
            mask_gt = mask_gt.squeeze_(1)
            mask_gt.requires_grad_(True)
            sparse_depth_gt = sparse_depth_gt.squeeze_(1)  # remove C dimension there's only one

            depth_loss = F.mse_loss(out*mask_gt, sparse_depth_gt*mask_gt)

            # Semanti Segmentation 
            # semantic_masks = list(
            #     map(lambda ann: ann['semantic_mask'], anns))
            # semantic_masks = tensorize_batch(
            #     semantic_masks, temp_variables.DEVICE)

            semantic_loss = F.cross_entropy(
                semantic_logits, semantic_masks.long())


            losses = {"depth_loss": depth_loss, "semantic_loss": semantic_loss, "loss_sum": depth_loss + semantic_loss}


            return losses

        else:

            return [{'semantic_logits': semantic_logits[idx], 'depth': out[idx]} for idx, _ in enumerate(img)]
        #
