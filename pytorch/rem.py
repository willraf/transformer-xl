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
        
    # def get_sinusoid(self, L, theta):

    #     k1, k2, k3, k4, k5, k6 = self.k1, self.k2, self.k3, self.k4, self.k5, self.k6
    #     M = L * theta
    #     s1 = torch.cos(M[:k2, ])
    #     s2 = torch.sin(M[k2:(k2+k3), ])
    #     s3 = torch.cos(M[(k2+k3):(k2+k3+k4), ])
    #     s4 = torch.sin(M[(k2+k3+k4):, ])
    #     s = torch.cat([s1,s2,s3,s4])
    #     return s

    def get_sinusoid(self, L, theta):
        k1, k2, k3, k4, k5, k6 = self.k1, self.k2, self.k3, self.k4, self.k5, self.k6
        M = L * theta
        s1 = torch.cos(M[:k2, ])
        s2 = torch.sin(M[k2:(k2+k3), ])
        s3 = torch.cos(M[(k2+k3):(k2+k3+k5), ]) #!
        s4 = torch.sin(M[(k2+k3+k5):(k2+k3+k5+k6), ]) #!
        s = torch.cat([s1,s2,s3,s4])
        return s.to(dtype=torch.float32, device=self.device)

    def forward(self, eta, nu, theta, query_len, key_len):
        lambda_ = torch.tanh(eta)
        gamma = torch.sigmoid(nu)
        L = self.create_Toeplitz_3D(self.d, self.truncation, query_len, key_len) # L is of shape (n_heads x query_len x key_len)
        # L1 = L[:2, :, :]
        # L2 = L[2:, :, :]
        # # s1,s2,s3,s4 = get_sinusoid(L,theta)
        # s = self.get_sinusoid(L2, theta)
        # powered_lambda = pow(lambda_,L1)
        # powered_gamma = pow(gamma,L2)
        # REM = torch.cat([powered_lambda, (powered_gamma * s)])

        # s1,s2,s3,s4 = get_sinusoid(L,theta)
        s = self.get_sinusoid(L, theta)
        powered_gamma = pow(gamma,L)
        REM = powered_gamma * s   

        return REM.permute(1, 2, 0)      # query_len x key_len x n_heads

    def create_Toeplitz_3D(self, d, truncation, query_len, key_len):
        x = np.arange(0, key_len)
        y = np.arange(0, query_len)

        T = torch.tensor(toeplitz(y, x))
        T[T > 200] = 0
        L = T.unsqueeze(0).repeat(self.n_head, 1, 1)

        d = torch.tensor(d).view(self.n_head, 1, 1)
        # d = d.to(dtype=torch.float32, device=self.device)
        # L = L.to(dtype=torch.float32, device=self.device)

        result_tensor = torch.empty_like(L)
        # Apply the function to each L matrix with the corresponding d value
        for i in range(self.n_head):
            result_tensor[i] = self.compute_Ld(T[i], d[i])
        return result_tensor.to(dtype=torch.float32, device=self.device)


    def compute_Ld(self, L, d):
        # Compute the indicator matrix: 1 where L is divisible by d, else 0
        indicator_matrix = (L % d == 0).int() 
        indicator_matrix = indicator_matrix.to(dtype=torch.float32, device=self.device)
        d = d.to(dtype=torch.float32, device=self.device)
        L = L.to(dtype=torch.float32, device=self.device)

        
        # Compute the result matrix L_d by element-wise division where the indicator is 1
        L_d = (L / d) * indicator_matrix
        return L_d


# import numpy as np
# import torch
# from scipy.linalg import toeplitz
# import torch.nn as nn


# class REM(nn.Module):
#     # initialise the object
#     def __init__(self, k1, k2, k3, k4, k5, k6, d, truncation, device, n_head):
#         super(REM, self).__init__()
        
#         self.k1 = k1 #(reg)
#         self.k2 = k2 #(c1)
#         self.k3 = k3 #(c2)
#         # dilated versions
#         self.k4 = k4
#         self.k5 = k5
#         self.k6 = k6
        
#         self.d = d
#         self.truncation = truncation
        
#         self.device = device
        
  
#     def get_sinusoid(self, L, theta):
#         M = L * theta
#         s1 = torch.cos(M[:self.k2, ]).to(self.device) 
#         s2 = torch.sin(M[self.k2:(self.k2+self.k3), ]).to(self.device) 
#         s3 = torch.cos(M[(self.k2+self.k3):(self.k2+self.k3+self.k5), ]) #!
#         s4 = torch.sin(M[(self.k2+self.k3+self.k5):(self.k2+self.k3+self.k5+self.k6), ]) #!
#         s = torch.cat([s1,s2,s3,s4]).to(self.device)
#         return s
        
#     def forward(self, eta, nu, theta, query_len, key_len):
#         # print(f'k1: {self.k1}, k2: {self.k2}, k3: {self.k3}, k4: {self.k4}, k5: {self.k5}, k6: {self.k6} ')
#         lambda_ = torch.tanh(eta).to(self.device)
#         gamma = torch.sigmoid(nu).to(self.device)
#         L = self.create_Toeplitz_3D(self.d, self.truncation, query_len, key_len) # L is of shape (n_heads x query_len x key_len)
        
#         powered_lambda = pow(lambda_,L)
#         powered_gamma = pow(gamma, L) 
#         s = self.get_sinusoid(L, theta) #cyclics
        
#         powered_gamma = powered_gamma[:self.k2 + self.k3 + self.k5 + self.k6] #!
#         powered_lambda = powered_lambda[:self.k1 + self.k4] #!
        
#         # initialize with just the regulars and then add the dilated
#         regular_rems = powered_lambda[:self.k1]
#         cyclic_rems = powered_gamma[:self.k2 + self.k3]
                
#         # dilate regular rems: (k4)
#         n_dilated_regs = self.k4
#         for i in range(n_dilated_regs):
#             dilated_reg_rem = torch.kron(powered_lambda[self.k1 + i], torch.eye(n=self.d.pop()).to(self.device)).to(self.device)
#             dilated_reg_rem = dilated_reg_rem[:L.shape[1], :L.shape[2]]
#             regular_rems = torch.cat([regular_rems, torch.unsqueeze(dilated_reg_rem, 0).to(self.device)]).to(self.device)

#         # dilate cyclic rems: (k5, k6)
#         n_dilated_cyclics = self.k5 + self.k6
#         for j in range(n_dilated_cyclics):
#             dilated_cyclic_rem = torch.kron(s[self.k2+self.k3+j], torch.eye(n=self.d.pop()).to(self.device)).to(self.device)
#             dilated_cyclic_rem = dilated_cyclic_rem[:L.shape[1], :L.shape[2]]        
#             cyclic_rems = torch.cat([cyclic_rems, torch.unsqueeze(dilated_cyclic_rem, 0).to(self.device)]).to(self.device)
        
#         # mask & -I
#         REM = torch.cat([regular_rems, (cyclic_rems * s)]).to(self.device)
#         REM = torch.tril(REM).to(self.device) - torch.eye(n=REM.shape[1], m=REM.shape[2]).to(self.device)
#         return REM

#     def create_Toeplitz_3D(self, d, truncation, query_len, key_len):
#         x = np.arange(0, key_len)
#         y = np.arange(0, query_len)

#         A = toeplitz(y, x)
#         A[A > 200] = 0
#         L = torch.from_numpy(A).to(self.device)
#         L = torch.stack([L]*4, 0).to(self.device)
#         return L.to(dtype=torch.float32, device=self.device)
