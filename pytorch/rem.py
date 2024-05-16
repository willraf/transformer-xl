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


    # WILLS
    def get_sinusoid(self, L, theta):

        k1, k2, k3, k4, k5, k6 = self.k1, self.k2, self.k3, self.k4, self.k5, self.k6
        M = L * theta
        s2 = torch.cos(M[k1:(k1+k2), ]).to(dtype=torch.float32, device=self.device)
        s3 = torch.sin(M[k2:(k2+k3), ]).to(dtype=torch.float32, device=self.device).to(dtype=torch.float32, device=self.device)
        s5 = torch.cos(M[(k2+k3+k4):(k2+k3+k4+k5), ]).to(dtype=torch.float32, device=self.device)
        s6 = torch.sin(M[(k2+k3+k4+k5):, ]).to(dtype=torch.float32, device=self.device)
        # s = torch.cat([s2,s3,s5,s6])
        return s2,s3,s5,s6

    # def forward(self, eta, nu, theta, query_len, key_len):
    #     lambda_ = torch.tanh(eta)
    #     gamma = torch.sigmoid(nu)
    #     L = self.create_Toeplitz_3D(self.d, self.truncation, query_len, key_len) # L is of shape (n_heads x query_len x key_len)
    #     # L1 = L[:2, :, :]
    #     # L2 = L[2:, :, :]
    #     # # s1,s2,s3,s4 = get_sinusoid(L,theta)
    #     # s = self.get_sinusoid(L2, theta)
    #     # powered_lambda = pow(lambda_,L1)
    #     # powered_gamma = pow(gamma,L2)
    #     # REM = torch.cat([powered_lambda, (powered_gamma * s)])

    #     # s1,s2,s3,s4 = get_sinusoid(L,theta)
    #     s = self.get_sinusoid(L, theta)
    #     powered_gamma = pow(gamma,L)
    #     REM = powered_gamma * s   

    def forward(self, eta, nu, theta, query_len, key_len):
        lambda_ = torch.tanh(eta)
        gamma = torch.sigmoid(nu)
        L = self.create_Toeplitz_3D(self.d, self.truncation, query_len, key_len) # L is of shape (n_heads x query_len x key_len)
        
        k1, k2, k3, k4, k5, k6 = self.k1, self.k2, self.k3, self.k4, self.k5, self.k6

        # Rems 2 3 5 and 6 are cyclic
        s2,s3,s5,s6 = self.get_sinusoid(L, theta)

        # Divide the L matix into the different rem types
        L1 = L[:k1]
        L2 = L[k1:(k1+k2)]
        L3 = L[(k1+k2):(k1+k2+k3)]
        L4 = L[(k1+k2+k3):(k1+k2+k3+k4)]
        L5 = L[(k1+k2+k3+k4):(k1+k2+k3+k4+k5)]
        L6 = L[(k1+k2+k3+k4+k5):]

        # # Regular (non cyclic) REMs
        P1 = torch.pow(lambda_,L1)
        P4 = torch.pow(lambda_,L4)

        # # Cyclic REMs
        P2 = torch.pow(gamma,L2) * s2
        P3 = torch.pow(gamma,L3) * s3
        P5 = torch.pow(gamma,L5) * s5
        P6 = torch.pow(gamma,L6) * s6

        REM = torch.cat([L1, L2, L3, L4, L5, L6])

        return REM.permute(1, 2, 0)      # query_len x key_len x n_heads

    def create_Toeplitz_3D(self, d, truncation, query_len, key_len):
        x = np.arange(0, key_len)
        y = np.arange(0, query_len)

        T = torch.tensor(toeplitz(y, x))
        T[T > 200] = 0
        L = T.unsqueeze(0).repeat(self.n_head, 1, 1)

        # Distil the final n rems (k4, k5, k6)
        n_distil = len(d)
        n_reg = self.n_head - n_distil
        d = torch.tensor(d).view(n_distil, 1, 1)

        result_tensor = torch.empty_like(L)
        # Apply the function to the distilled REMs
        for i in range(n_distil):
            result_tensor[n_reg + i] = self.compute_Ld(L[n_reg + i], d[i])
        return result_tensor.to(dtype=torch.float32, device=self.device)


    def compute_Ld(self, L, d):
        # Compute the indicator matrix: 1 where L is divisible by d, else 0
        indicator_matrix = (L % d == 0).int() 
        indicator_matrix = indicator_matrix.to(dtype=torch.float32, device=self.device)
        d = torch.tensor(d, dtype=torch.float32, device=self.device)
        L = L.to(dtype=torch.float32, device=self.device)

        # Compute the result matrix L_d by element-wise division where the indicator is 1
        L_d = (L / d) * indicator_matrix
        return L_d