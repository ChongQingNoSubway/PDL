import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.nystrom_attention import NystromAttention
from models.dropout import Pdropout

class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, input_size, n_classes, mDim=512):
        super(TransMIL, self).__init__()
        self.pos_layer = PPEG(dim=mDim)
        #self._fc1 = nn.Sequential(nn.Linear(input_size, mDim), nn.ReLU(), nn.Dropout(0.2))
        self._fc1 = nn.Sequential(
          nn.Linear(input_size, 256),
          nn.ReLU(),
          Pdropout(0.35),
          nn.Linear(256, 128),
          nn.ReLU(),
          Pdropout(0.35),
          nn.Linear(128, mDim),
          nn.ReLU(),
          Pdropout(0.35)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, mDim))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=mDim)
        self.layer2 = TransLayer(dim=mDim)
        self.norm = nn.LayerNorm(mDim)
        self._fc2 = nn.Linear(mDim, self.n_classes)


    def forward(self, **kwargs):
        h = kwargs['data'].float() #[B, n, 1024]
        
        #h = self._fc1(h) #[B, n, 512]
        h = self._fc1(h.squeeze(0)).unsqueeze(0)
        #h = self.sc_layer(h.squeeze(0)).unsqueeze(0)
        
        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1
        h = self.layer1(h) #[B, N, 512]

        #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N, 512]
        
        #---->Translayer x2
        h = self.layer2(h) #[B, N, 512]

        #---->cls_token
        h = self.norm(h)[:,0]

        #---->predict
        logits = self._fc2(h) #[B, n_classes]
       
        return logits, h

if __name__ == "__main__":
    data = torch.randn((1, 6000, 512))
    model = TransMIL(512, n_classes=2, mDim=484)
    print(model.eval())
    results_dict = model(data = data)
    print(results_dict)