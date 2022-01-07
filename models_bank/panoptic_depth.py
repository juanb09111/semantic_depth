
import torch
from torch import nn
import torch.nn.functional as F
import torchvision

from utils.tensorize_batch import tensorize_batch
from segmentation_heads.sem_seg import segmentation_head as sem_seg_head

from models_bank.maskrcnn.detection import maskrcnn_resnet50_fpn
from models_bank.maskrcnn.detection.backbone_utils import resnet_fpn_backbone


from common_blocks.depth_wise_conv import depth_wise_conv
from common_blocks.depth_wise_sep_conv import depth_wise_sep_conv
from common_blocks.continuous_conv import ContinuousConvolution
from segmentation_heads.refine_head import refine_head
import config_kitti


# %%


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


class PanopticDepth(nn.Module):
    def __init__(self, k_number,
                 backbone_out_channels,
                 num_ins_classes,
                 num_sem_classes,
                 input_image_size,
                 n_number=None,
                 pre_trained_backboned=False,
                 backbone_name="resnet50"):

        super(PanopticDepth, self).__init__()

        # Depth completion ------------------

        # Depth head---------------------------------------------------------------

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
            nn.Conv2d(in_channels=4 + num_ins_classes + num_sem_classes + 1, out_channels=32,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.fuse_conv = nn.Sequential(
            FuseBlock(48, 64, k_number, n_number=n_number),
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
        min_size, max_size = input_image_size

        self.backbone = resnet_fpn_backbone(
            backbone_name, pre_trained_backboned)
        # backbone.body.conv1 = nn.Conv2d(48, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.mask_rcnn = maskrcnn_resnet50_fpn(
            pretrained=False, backbone=self.backbone, num_classes=num_ins_classes + 1, min_size=min_size, max_size=max_size)

        self.semantic_head = sem_seg_head(
            backbone_out_channels,
            num_ins_classes + num_sem_classes + 1, input_image_size, depthwise_conv=config_kitti.SEMANTIC_HEAD_DEPTHWISE_CONV)

        # output

        self.refine_head = refine_head(
            num_ins_classes + num_sem_classes + 1, input_image_size)

        for module in self.children():
            if self.training:
                module.training = True
            else:
                module.training = False

    def to(self, device):
        for module in self.children():
            module.to(device)

    def forward(self,
                images,
                sparse_depth,
                mask,
                coors,
                k_nn_indices,
                sparse_depth_gt=None,
                anns=None):

        # losses = {}
        semantic_logits = []

        if self.training:
            maskrcnn_losses, backbone_feat = self.mask_rcnn(images, anns)

        else:
            maskrcnn_results, backbone_feat = self.mask_rcnn(images)

        P4, P8, P16, P32 = backbone_feat['0'], backbone_feat['1'], backbone_feat['2'], backbone_feat['3']

        semantic_logits = self.semantic_head(P4, P8, P16, P32)

        # Depth completion ------------------------
        _, H, W = mask.shape

        # sparse depth branch
        y_sparse = self.sparse_conv(sparse_depth)  # B x 16 x H/2 x W/2

        # rgbd branch
        x_concat_d = torch.cat((images, sparse_depth, semantic_logits), dim=1)
        y_rgbd = self.rgbd_conv(x_concat_d)  # B x 32 x H/2 x W/2

        y_rgbd_cat_y_sparse = torch.cat((y_rgbd, y_sparse), dim=1)

        y_rgbd_cat_y_sparse = F.interpolate(y_rgbd_cat_y_sparse, (H, W))

        fused, _, _, _ = self.fuse_conv(
            (y_rgbd_cat_y_sparse, mask, coors, k_nn_indices))

        fused_out = self.output_layer(fused)
        # output

        semantic_logits = self.refine_head(semantic_logits, fused_out)

        # out = fused_out.squeeze_(1)
        out = torch.squeeze(fused_out, 1)
        if self.training:

            device = semantic_logits.get_device()

            # Depth completion
            mask_gt = torch.where(sparse_depth_gt > 0,
                                  torch.tensor((1), device=device, dtype=torch.float64),
                                  torch.tensor((0), device=device, dtype=torch.float64))
            mask_gt = mask_gt.squeeze_(1)
            mask_gt.requires_grad_(True)
            sparse_depth_gt = sparse_depth_gt.squeeze_(
                1)  # remove C dimension there's only one

            depth_loss = F.mse_loss(out*mask_gt, sparse_depth_gt*mask_gt)

            semantic_masks = list(map(lambda ann: ann['semantic_mask'], anns))
            semantic_masks = tensorize_batch(semantic_masks, device)

            semantic_loss = F.cross_entropy(
                semantic_logits, semantic_masks.long())

            return {"depth_loss": depth_loss, "semantic_loss": semantic_loss, **maskrcnn_losses, "loss_sum": depth_loss + semantic_loss}

        else:
            return [{**maskrcnn_results[idx], 'semantic_logits': semantic_logits[idx], 'depth': out[idx]} for idx, _ in enumerate(images)]
