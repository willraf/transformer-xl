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
        self.d = torch.tensor(d, dtype=torch.float32)
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
        s2 = torch.cos(M[k1:(k1+k2), ])
        s3 = torch.sin(M[k2:(k2+k3), ])
        s5 = torch.cos(M[(k2+k3+k4):(k2+k3+k4+k5), ])
        s6 = torch.sin(M[(k2+k3+k4+k5):, ])
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
        # L_non_dil = L[:k3, :, :]
        # L_dil = L[k3:, :, :]

        # result_L_dil = torch.empty_like(L_dil)
        # k4 5 and 6 are diluted
        undil_n = k1+k2+k3
        dil_n = k4 + k5 + k6
        for i in range(undil_n, dil_n): 
          L[i] = self.compute_Ld(L[i], self.d[i])

        L = L.to(dtype=torch.float32, device=self.device)

        # Rems 2 3 5 and 6 are cyclic
        s2,s3,s5,s6 = self.get_sinusoid(L,theta)

        L1 = L[:k1]
        L2 = L[k1:(k1+k2)]
        L3 = L[(k1+k2):(k1+k2+k3)]
        L4 = L[(k1+k2+k3):(k1+k2+k3+k4)]
        L5 = L[(k1+k2+k3+k4):(k1+k2+k3+k4+k5)]
        L6 = L[(k1+k2+k3+k4+k5):]

        # Regular (non cyclic) REMs
        L1 = pow(lambda_,L1)
        L4 = pow(lambda_,L4)

        # Cyclic REMs
        L2 = pow(gamma,L2) * s2
        L3 = pow(gamma,L3) * s3
        L5 = pow(gamma,L5) * s5
        L6 = pow(gamma,L6) * s6

        REM = torch.cat([L1, L2, L3, L4, L5, L6])

        # s = self.get_sinusoid(L, theta)
        # powered_lambda = pow(lambda_,L1)
        # powered_gamma = pow(gamma,L2)
        # REM = torch.cat([powered_lambda, (powered_gamma * s)])



        # s1,s2,s3,s4 = get_sinusoid(L,theta)
        # s = self.get_sinusoid(L, theta)
        # powered_gamma = pow(gamma,L)
        # REM = powered_gamma * s   

        return REM.permute(1, 2, 0)      # query_len x key_len x n_heads

    def create_Toeplitz_3D(self, d, truncation, query_len, key_len):
        x = np.arange(0, key_len)
        y = np.arange(0, query_len)

        T = torch.tensor(toeplitz(y, x), dtype=torch.float32)
        T[T > 200] = 0
        L = T.unsqueeze(0).repeat(self.n_head, 1, 1)
        return L

    # def create_Toeplitz_3D(self, d, truncation, query_len, key_len):
    #     x = np.arange(0, key_len)
    #     y = np.arange(0, query_len)

    #     T = torch.tensor(toeplitz(y, x))
    #     T[T > 200] = 0
    #     L = T.unsqueeze(0).repeat(self.n_head, 1, 1)

    #     d = torch.tensor(d).view(self.n_head, 1, 1)
    #     # d = d.to(dtype=torch.float32, device=self.device)
    #     # L = L.to(dtype=torch.float32, device=self.device)

    #     result_tensor = torch.empty_like(L)
    #     # Apply the function to each L matrix with the corresponding d value
    #     for i in range(self.n_head):
    #         result_tensor[i] = self.compute_Ld(T[i], d[i])
    #     return result_tensor.to(dtype=torch.float32, device=self.device)


    def compute_Ld(self, L, d):
        # Compute the indicator matrix: 1 where L is divisible by d, else 0
        indicator_matrix = (L % d == 0)
        # indicator_matrix = indicator_matrix.to(dtype=torch.float32, device=self.device)
        # L = L.to(dtype=torch.float32, device=self.device)
        
        # Compute the result matrix L_d by element-wise division where the indicator is 1
        L_d = (L / d) * indicator_matrix
        return L_d