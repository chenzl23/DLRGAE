import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch


class DLRGAE(nn.Module):
    def __init__(self, args, dim_in, dim_out, N):
        super(DLRGAE, self).__init__()
        self.args = args
        self.dropout_rate = args.dropout
        self.N = N

        self.dim_in = dim_in
        self.dim_out = dim_out

        self.U_number = 5

        self.relu = nn.ReLU(inplace=True)

        self.setup_layers(args)

    def setup_layers(self, args):
        """
        Creating the layes based on the args.
        """
        # set layer weights for GCN
        self.args.layers = [args.hidden_dim for i in range(args.layer_num - 1)]
        self.args.layers = [self.dim_in] + self.args.layers + [self.dim_out]

        self.layers = nn.ModuleList()

        for i, _ in enumerate(self.args.layers[:-1]):
            self.layers.append(GCNConv(self.args.layers[i], self.args.layers[i+1], normalize=False))

        self.U = nn.ParameterList()
        for i in range(self.U_number):
            self.U.append(nn.Parameter(torch.randn(self.dim_out, self.dim_out)))


    def projection_p(self, P):
        dim = P.shape[0]
        one = torch.ones(dim, 1).to(self.args.device)

        relu = torch.nn.ReLU()
        '''
        Projection
        '''
        P_1 = relu(P) 

        support_1 = torch.mm(torch.mm(P_1, one) - one, one.t()) / dim
        P_2 = P_1 - support_1

        support_2 = torch.mm(one, torch.mm(one.t(), P_2) - one.t()) / dim
        P_3 = P_2 - support_2

        return P_3
    
    def softmax_projection(self, x):
        sm_0 = torch.nn.Softmax(dim = 0)
        sm_1 = torch.nn.Softmax(dim = 1)

        x_0 = sm_0(x)
        x_1 = sm_1(x)

        proj_x = (x_0 + x_1) / 2
        return proj_x

    def multi_projection(self, x):
        proj_x = self.softmax_projection(x)
        for i in range(10):
            proj_x = self.projection_p(proj_x)

        return proj_x


    def forward(self, graph):
        Z = graph.x
        Z_knn = graph.x
        A = graph.adj
        A_knn = graph.adj_knn

        ## Compute common U
        Us = self.U[0]
        for i in range(1, self.U_number):
            Us = Us.matmul(self.U[i])
        self.Us = self.multi_projection(Us.matmul(Us.t()))

        ## topology graph
        for i in range(len(self.layers) - 1):
            Z = self.relu(self.layers[i](Z, A))
            Z = F.dropout(Z, p = self.dropout_rate, training=self.training)
        Z = self.layers[-1](Z, A)

        Z_emb = F.softmax(Z)

        
        estimated_A = Z_emb.matmul(self.Us).matmul(Z_emb.t())
        estimated_A = torch.sigmoid(estimated_A)


        ## knn graph
        for i in range(len(self.layers) - 1):
            Z_knn = self.relu(self.layers[i](Z_knn, A_knn))
            Z_knn = F.dropout(Z_knn, training=self.training)
        Z_knn = self.layers[-1](Z_knn, A_knn)

        Z_emb_knn = F.softmax(Z_knn)
        
        estimated_A_knn = Z_emb_knn.matmul(self.Us).matmul(Z_emb_knn.t())
        estimated_A_knn = torch.sigmoid(estimated_A_knn)

        return Z, Z_knn, estimated_A, estimated_A_knn