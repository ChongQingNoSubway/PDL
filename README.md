# PDL: Regularizing Multiple Instance Learning with Progressive Dropout Layers

### [PDL: Regularizing Multiple Instance Learning with Progressive Dropout Layers](https://arxiv.org/pdf/2308.10112.pdf)

## Reference MIL methods github repo: [DSMIL](https://github.com/binli123/dsmil-wsi)   [TransMIl](https://github.com/szc19990412/TransMIL) [DTFD-MIL](https://github.com/hrzhang1123/DTFD-MIL) [ABMIL](https://github.com/AMLab-Amsterdam/AttentionDeepMIL)

**Abstract** 
Multiple instance learning (MIL) was a weakly supervised learning approach that sought to assign binary class labels to collections of instances known as bags. However, due to their weak supervision nature, the MIL methods were susceptible to overfitting and required assistance in developing comprehensive representations of target instances. While regularization typically effectively combated overfitting, its integration with the MIL model has been frequently overlooked in prior studies. Meanwhile, current regularization methods for MIL have shown limitations in their capacity to uncover a diverse array of representations. In this study, we delve into the realm of regularization within the MIL model, presenting a novel approach in the form of a Progressive Dropout Layer (PDL). We aim to not only address overfitting but also empower the MIL model in uncovering intricate and impactful feature representations. The proposed method was orthogonal to existing MIL methods and could be easily integrated into them to boost performance. Our extensive evaluation across a range of MIL benchmark datasets demonstrated that the incorporation of the PDL into multiple MIL methods not only elevated their classification performance but also augmented their potential for weakly-supervised feature localizations.

**Framework** 
<img src="images/3.png"/>

**Experiment Result**
<img src="images/2.png"/>
<img src="images/1.png"/>
<img src="images/4.png"/>
<img src="images/5.png"/>
## How to use
### In Network.py
```
from models.dropout import Pdropout
.......
class TransMIL(nn.Module):
    def __init__(self, input_size, n_classes, mDim=512):
        super(TransMIL, self).__init__()
        self.pos_layer = PPEG(dim=mDim)
        self._fc1 = nn.Sequential(nn.Linear(input_size, mDim), nn.ReLU(), nn.Dropout(0.2))
        # the fc1 as the middle layer
        self._fc1 = nn.Sequential(
          nn.Linear(input_size, 256),
          nn.ReLU(),
          # append the PDL after Relu activation function
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
........
```
### IN train.py
```
from models.dropout import LinearScheduler
.......
milnet = TransMIL(args.feats_size, args.num_classes, mDim=args.L).cuda()
progressive_schedule = LinearScheduler(milnet,start_value=0,stop_value=P_max,nr_steps=args.num_epochs)

for epoch in range(1, args.num_epochs + 1):
        progressive_schedule.step()
........
```
