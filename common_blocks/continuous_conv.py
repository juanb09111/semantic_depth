
import torch
import torch.nn as nn
import temp_variables
class ContinuousConvolution(nn.Module):
    """
    http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Deep_Parametric_Continuous_CVPR_2018_paper.pdf
    h_i = W(Sum_k MLP(x_i - x_k) * f_k)
    inputs:
    + x: B x N x C (points features)
    + points: B x N x 3 (points coordinates)
    + indices: B x N x K (knn indices)

    outputs:
    + y: points features
    """

    def __init__(self, n_feat, k_number, n_number=None):
        super().__init__()

        self.linear1=nn.Linear(3 * k_number, (n_feat // 2) * k_number)  # B x N x 3*(n_feat//2)
        self.batch_norm1=nn.BatchNorm1d(n_number, track_running_stats=False)  # B x N(normalize this dim) x 3*(n_feat//2)
        self.relu1=nn.ReLU()

        self.linear2=nn.Linear((n_feat // 2) * k_number, n_feat * k_number)  # B x N x n_feat*k_number
        self.batch_norm2=nn.BatchNorm1d(n_number, track_running_stats=False)  # B x N(again, norm this dim) x n_feat*k_number
        self.relu2=nn.ReLU()


        # self.mlp = nn.Sequential(
        #     # input: B x N x 3 x K
        #     nn.Linear(3 * k_number, (n_feat // 2) * k_number),  # B x N x 3*(n_feat//2)
        #     nn.BatchNorm1d(n_number, track_running_stats=False),  # B x N(normalize this dim) x 3*(n_feat//2)
        #     nn.ReLU(),
        #     nn.Linear((n_feat // 2) * k_number, n_feat * k_number),  # B x N x n_feat*k_number
        #     nn.BatchNorm1d(n_number, track_running_stats=False),  # B x N(again, norm this dim) x n_feat*k_number
        #     nn.ReLU()
        # )

        # self.mlp_eval = nn.Sequential(
        #     # input: B x N x 3 x K
        #     nn.Linear(3 * k_number, (n_feat // 2) * k_number),  # B x N x 3*(n_feat//2)
        #     nn.ReLU(),
        #     nn.Linear((n_feat // 2) * k_number, n_feat * k_number),  # B x N x n_feat*k_number
        #     nn.ReLU()
        # )

    def forward(self, *inputs):
        """
        NOTE: *inputs indicates the input parameter expect a tuple
             **inputs indicates the input parameter expect a dict
        x: B x N x C (points features)
        points: B x N x 3 (points coordinates)
        indices: B x N x K (knn indices)
        """
        x, points, indices = inputs[0]
        B, N, C = x.size()
        K = indices.size(2)

        y1 = torch.cat([points[i, indices[i]] for i in range(B)], dim=0)
        y1 = y1.view(B,N,K,3)  # B x N x K x 3
        y1 = points[:, :, None, :] - y1  # B x N x K x 3
        y1 = y1.view(B, N, K * 3)   # B x N x K*3 
        
        if self.training:
            # y1 = self.mlp(y1).view(B, N, K, C)  # reshape after mlp B x N x K x C

            # ----
            y1 = self.linear1(y1)
            y1 = self.batch_norm1(y1)
            y1 = self.relu1(y1)

            y1 = self.linear2(y1)
            y1 = self.batch_norm2(y1)
            y1 = self.relu2(y1).view(B, N, K, C)  # reshape after mlp B x N x K x C

        if not self.training:
            # y1 = self.mlp_eval(y1).view(B, N, K, C)  # reshape after mlp B x N x K x C 

            y1 = self.linear1(y1)
            y1 = self.relu1(y1)

            y1 = self.linear2(y1)
            y1 = self.relu2(y1).view(B, N, K, C)  # reshape after mlp B x N x K x C

        y2 = torch.cat([x[i, indices[i]] for i in range(B)], dim=0).view(B, N, K, C)  # B x N x K x C
        # print("y2 ,,", y2.shape)
        return torch.sum(y1 * y2, dim=2), points, indices