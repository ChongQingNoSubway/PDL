import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dropout import Pdropout
from models.nl import NLBlockND



class GatedAttention(nn.Module):
    def __init__(self,in_features,num_class,drop_p):
        super(GatedAttention, self).__init__()
        self.L = 64
        self.D = 64
        self.K = 1

        self.feature_extractor = nn.Sequential(
          nn.Linear(in_features, 256),
          nn.ReLU(),
          nn.Linear(256,128),
          nn.ReLU(),
          nn.Linear(128, self.L),
          nn.ReLU(),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid(),
            
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, num_class),
        )

    def forward(self, x):
        x = x.squeeze(0)

        #H = self.feature_extractor_part1(x)
        #H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor(x)  # NxL

        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        #Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, A

class AttnMIL(nn.Module):
    def __init__(self, in_features,num_class,drop_p):
        super().__init__()
        self.L = 64
        self.D = 64
        self.K = 1

        self.feature_extractor = nn.Sequential(
          nn.Linear(in_features, 256),
          nn.ReLU(),
          Pdropout(0.2),

          nn.Linear(256, 128),
          nn.ReLU(),
          Pdropout(0.2),

          nn.Linear(128, self.L),
          nn.ReLU(),
          Pdropout(0.2),
        )

        self.attn = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            #nn.Dropout1d(drop_p),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, num_class),
            #nn.Sigmoid()
        )


    def forward(self, x):
        x = x.squeeze(0)
        H = self.feature_extractor(x) # NxL

        A = self.attn(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        #Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, A



class AttnMIL_pdl(nn.Module):
    def __init__(self, in_features,num_class,drop_p):
        super().__init__()
        self.L = 64
        self.D = 64
        self.K = 1

        self.feature_extractor = nn.Sequential(
          nn.Linear(in_features, 256),
          nn.ReLU(),
          Pdropout(0.2),

          nn.Linear(256, 128),
          nn.ReLU(),
          Pdropout(0.2),

          nn.Linear(128, self.L),
          nn.ReLU(),
          Pdropout(0.2),
        )

        self.attn = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, num_class),
            #nn.Sigmoid()
        )


    def forward(self, x):
        #print(x.shape)
        #x = x.squeeze(0)
        H = self.feature_extractor(x) # NxL

        A = self.attn(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        #Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, A
    


class Pdrop_GatedAttention(nn.Module):
    def __init__(self,in_features,num_class,drop_p):
        super(Pdrop_GatedAttention, self).__init__()
        self.L = 64
        self.D = 64
        self.K = 1

        self.feature_extractor = nn.Sequential(
          nn.Linear(in_features, 256),
          nn.ReLU(),
          Pdropout(drop_p),
          nn.Linear(256, 128),
          nn.ReLU(),
          Pdropout(drop_p),
          nn.Linear(128, self.L),
          nn.ReLU(),
          Pdropout(drop_p)
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            #nn.Dropout1d(drop_p),
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid(),
            #nn.Dropout1d(drop_p),
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, num_class),
            #nn.Sigmoid()
        )

    def forward(self, x):
        #x = x.squeeze(0)

        #H = self.feature_extractor_part1(x)
        #H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor(x)  # NxL

        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        #Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, A


    