import torch
import torch.nn as nn
import torch.nn.functional as F
from pdl import Pdropout

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.L = 256
        self.D = 128
        self.K = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        # self.feature_extractor_part2 = nn.Sequential(
        #     nn.Linear(50 * 4 * 4, 512),
        #     nn.ReLU(),
        #     Pdropout(),
        #     nn.Linear(50 * 4 * 4, 512),
        #     nn.ReLU(),
        # )

        self.feature_ex2_0 =  nn.Linear(50 * 4 * 4, 512)
        self.feature_ex2_1 =  nn.ReLU()
        self.feature_ex2_2 =  Pdropout(0.35)
        self.feature_ex2_3 =  nn.Linear(512, self.L)
        self.feature_ex2_4 =  nn.ReLU()
        self.feature_ex2_5 =  Pdropout(0.35)
        
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1)
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 4 * 4)
        #H = self.feature_extractor_part2(H)  # NxL
        
        #feature extractor2 
        H = self.feature_ex2_0(H)
        H = self.feature_ex2_1(H)
        H,A_0 = self.feature_ex2_2(H)
        H = self.feature_ex2_3(H)
        H = self.feature_ex2_4(H)
        H,A_1 = self.feature_ex2_5(H)
        

        A = self.attention(H)  # NxK
        #A = torch.mean(H,dim=1,keepdim=True)

        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)

        return Y_prob, A

