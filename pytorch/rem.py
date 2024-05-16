import numpy as np
import torch
from scipy.linalg import toeplitz
import torch.nn as nn


class REM(nn.Module):
    # initialise the object
    def __init__(self, k1, k2, k3, k4, k5, k6, d, truncation, device=None, n_head=4):
        super(REM, self).__init__()
        
        self.k1 = k1 #(reg)
        self.k2 = k2 #(c1)
        self.k3 = k3 #(c2)
        # dilated versions
        self.k4 = k4
        self.k5 = k5
        self.k6 = k6
        
        self.truncation = truncation

        self.device = device
        self.d = d
        self.n_head = n_head
    
    # # ORIGINAL
    # def get_sinusoid(self, L, theta):

    #     k1, k2, k3, k4, k5, k6 = self.k1, self.k2, self.k3, self.k4, self.k5, self.k6
    #     M = L * theta
    #     s1 = torch.cos(M[:k2, ])
    #     s2 = torch.sin(M[k2:(k2+k3), ])
    #     s3 = torch.cos(M[(k2+k3):(k2+k3+k4), ])
    #     s4 = torch.sin(M[(k2+k3+k4):, ])
    #     s = torch.cat([s1,s2,s3,s4])
    #     return s

    # WILLS
    def get_sinusoid(self, L, theta):

        k1, k2, k3, k4, k5, k6 = self.k1, self.k2, self.k3, self.k4, self.k5, self.k6
        M = L * theta
        s2 = torch.cos(M[k1:(k1+k2), ]).to(dtype=torch.float32, device=self.device)
        s3 = torch.sin(M[k2:(k2+k3), ]).to(dtype=torch.float32, device=self.device).to(dtype=torch.float32, device=self.device)
        s5 = torch.cos(M[(k2+k3+k4):(k2+k3+k4+k5), ]).to(dtype=torch.float32, device=self.device)
        s6 = torch.sin(M[(k2+k3+k4+k5):, ]).to(dtype=torch.float32, device=self.device)
        return s2,s3,s5,s6

    # # RORYS
    # def get_sinusoid(self, L, theta):
    #     k1, k2, k3, k4, k5, k6 = self.k1, self.k2, self.k3, self.k4, self.k5, self.k6
    #     M = L * theta
    #     s1 = torch.cos(M[:k2, ])
    #     s2 = torch.sin(M[k2:(k2+k3), ])
    #     s3 = torch.cos(M[(k2+k3):(k2+k3+k5), ]) #!
    #     s4 = torch.sin(M[(k2+k3+k5):(k2+k3+k5+k6), ]) #!
    #     s = torch.cat([s1,s2,s3,s4])
    #     return s.to(dtype=torch.float32, device=self.device)

    def forward(self, eta, nu, theta, query_len, key_len):
        lambda_ = torch.tanh(eta)
        gamma = torch.sigmoid(nu)
        L = self.create_Toeplitz_3D(self.d, self.truncation, query_len, key_len) # L is of shape (n_heads x query_len x key_len)
        
        k1, k2, k3, k4, k5, k6 = self.k1, self.k2, self.k3, self.k4, self.k5, self.k6

        L_distiled = torch.empty_like(L)
        print('L_disl ', L_distiled.shape)
        undil_n = k1+k2+k3
        dil_n = k4+k5+k6
        L_distiled[:undil_n] = L[:undil_n]
        for i in range(dil_n):
            L_distiled[(i+undil_n)] = self.compute_Ld(L[(i+undil_n)], self.d[i])

        # Rems 2 3 5 and 6 are cyclic
        s2,s3,s5,s6 = self.get_sinusoid(L_distiled,theta)

        L1 = L_distiled[:k1]
        L2 = L_distiled[k1:(k1+k2)]
        L3 = L_distiled[(k1+k2):(k1+k2+k3)]
        L4 = L_distiled[(k1+k2+k3):(k1+k2+k3+k4)]
        L5 = L_distiled[(k1+k2+k3+k4):(k1+k2+k3+k4+k5)]
        L6 = L_distiled[(k1+k2+k3+k4+k5):]

        # Regular (non cyclic) REMs
        P1 = pow(lambda_,L1)
        P4 = pow(lambda_,L4)

        # Cyclic REMs
        P2 = pow(gamma,L2) * s2
        P3 = pow(gamma,L3) * s3
        P5 = pow(gamma,L5) * s5
        P6 = pow(gamma,L6) * s6
        print('L', L.shape)
        print('L distilled', L_distiled.shape)
        print('L5 ', P5.shape)
        print('L6 ', P6.shape)

        REM = torch.cat([P1, P2, P3, P4, P5, P6])
        print(REM.shape)

        return REM.permute(1, 2, 0)      # query_len x key_len x n_heads

    def create_Toeplitz_3D(self, d, truncation, query_len, key_len):
        x = np.arange(0, key_len)
        y = np.arange(0, query_len)

        T = torch.tensor(toeplitz(y, x), dtype=torch.float32)
        T[T > 200] = 0
        L = T.unsqueeze(0).repeat(self.n_head, 1, 1)
        return L.to(dtype=torch.float32, device=self.device)


    def compute_Ld(self, L, d):
        # Compute the indicator matrix: 1 where L is divisible by d, else 0
        # L = L.to(dtype=torch.float32, device=self.device)
        # d = d.to(dtype=torch.float32, device=self.device)
        indicator_matrix = (L % d == 0).int() 
        indicator_matrix = indicator_matrix.to(dtype=torch.float32, device=self.device)
        # 
        
        # Compute the result matrix L_d by element-wise division where the indicator is 1
        L_d = (L / d) * indicator_matrix
        return L_d