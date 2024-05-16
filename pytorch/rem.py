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
        
        self.d = d
        self.truncation = truncation

        self.device = device
        self.n_head = n_head
        
    def get_sinusoid(self, L, theta):

        k1, k2, k3, k4, k5, k6 = self.k1, self.k2, self.k3, self.k4, self.k5, self.k6
        M = L * theta
        s1 = torch.cos(M[:k2, ])
        s2 = torch.sin(M[k2:(k2+k3), ])
        s3 = torch.cos(M[(k2+k3):(k2+k3+k4), ])
        s4 = torch.sin(M[(k2+k3+k4):, ])
        s = torch.cat([s1,s2,s3,s4])
        return s

    def forward(self, eta, nu, theta, query_len, key_len):
        lambda_ = torch.tanh(eta)
        gamma = torch.sigmoid(nu)
        L = self.create_Toeplitz_3D(self.d, self.truncation, query_len, key_len) # L is of shape (n_heads x query_len x key_len)
        print('L shape: ', L.shape)
        # s1,s2,s3,s4 = get_sinusoid(L,theta)
        s = self.get_sinusoid(L, theta)
        powered_lambda = pow(lambda_,L)
        powered_gamma = pow(gamma,L)
        REM = torch.cat([powered_lambda, (powered_gamma * s)])
        print('REM shape: ', REM.shape)
        print('transpoase REM shape', REM.transpose(0, 2).shape)
        return REM.transpose(0, 2)      # query_len x key_len x n_heads

    # def create_Toeplitz_3D(self, d, truncation, query_len):
    #     T = np.arange(query_len) 
    #     A = toeplitz(c=T)
    #     A[A > 200] = 0
    #     L = torch.from_numpy(A).float()
    #     # L = L[:][:truncation] #! truncate?
    #     L = torch.stack([L]*4, 0)
    #     return L.to(self.device)

    def create_Toeplitz_3D(self, d, truncation, query_len, key_len):
        x = np.arange(0, (key_len - 1))
        y = np.arange(0, (query_len - 1))

        T = torch.tensor(toeplitz(y, x))
        L = T.unsqueeze(0).repeat(slef.n_head, 1, 1)

        d = torch.tensor(d).view(self.n_head, 1, 1)
        print("d", d.shape)
        print('l', L.shape)
        L = L/d
        return L.to(self.device)

        # # Initialize the Toeplitz matrix
        # L = np.zeros((n_heads, query_len, key_len, truncation))

        # # Construct the Toeplitz matrix
        # for k in range(truncation):
        #     # Construct a 2D Toeplitz matrix
        #     T = toeplitz(range(d - k, 2 * d - k), r=np.arange(d, 0, -1))
        #     # Expand the 2D Toeplitz matrix to the required shape and assign it to the k-th slice of L
        #     L[:, :, :, k] = np.tile(T, (n_heads, query_len // n_heads, key_len // n_heads, 1))

        # return L


