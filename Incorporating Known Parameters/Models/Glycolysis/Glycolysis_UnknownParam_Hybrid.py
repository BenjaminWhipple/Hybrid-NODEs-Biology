import torch
import torch.nn as nn

device = torch.device('cpu')

class Unknown_Params_HybridODE(nn.Module):
    def __init__(self, p0, structure):
        super(Unknown_Params_HybridODE, self).__init__()
        """
        self.paramsODE = nn.Parameter(torch.tensor(p0).to(device))
        self.J0 = self.paramsODE[0]     #mM min-1
        #self.k1 = self.paramsODE[1]     #mM-1 min-1
        self.k2 = self.paramsODE[2]     #mM min-1
        self.k3 = self.paramsODE[3]     #mM min-1
        self.k4 = self.paramsODE[4]     #mM min-1
        self.k5 = self.paramsODE[5]     #mM min-1
        self.k6 = self.paramsODE[6]     #mM min-1
        self.k = self.paramsODE[7]      #min-1
        self.kappa = self.paramsODE[8]  #min-1
        #self.q = self.paramsODE[9]
        #self.K1 = self.paramsODE[10]    #mM
        self.psi = self.paramsODE[11]
        self.N = self.paramsODE[12]     #mM
        self.A = self.paramsODE[13]     #mM
        """
        J0,k1,k2,k3,k4,k5,k6,k,kappa,q,K1,psi,N,A = p0
        self.J0 = nn.Parameter(torch.tensor(J0).to(device))
        #self.k1 = nn.Parameter(torch.tensor(k1).to(device))
        self.k2 = nn.Parameter(torch.tensor(k2).to(device))
        self.k3 = nn.Parameter(torch.tensor(k3).to(device))
        self.k4 = nn.Parameter(torch.tensor(k4).to(device))
        self.k5 = nn.Parameter(torch.tensor(k5).to(device))
        self.k6 = nn.Parameter(torch.tensor(k6).to(device))
        self.k = nn.Parameter(torch.tensor(k).to(device))
        self.kappa = nn.Parameter(torch.tensor(kappa).to(device))
        #self.q = nn.Parameter(torch.tensor(q).to(device))
        #self.K1 = nn.Parameter(torch.tensor(K1).to(device))
        self.psi = nn.Parameter(torch.tensor(psi).to(device))
        self.N = nn.Parameter(torch.tensor(N).to(device))
        self.A = nn.Parameter(torch.tensor(A).to(device))

        self.net = self.make_nn(structure)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        S1 = y.view(-1,7)[:,0]
        S2 = y.view(-1,7)[:,1]
        S3 = y.view(-1,7)[:,2]
        S4 = y.view(-1,7)[:,3]
        S5 = y.view(-1,7)[:,4]
        S6 = y.view(-1,7)[:,5]
        S7 = y.view(-1,7)[:,6]

        dS1 = self.J0 + 0. * S1 #for dimensions
        dS2 = - self.k2 * S2 * (self.N - S5) - self.k6 * S2 * S5
        dS3 = self.k2 * S2 * (self.N - S5) - self.k3 * S3 * (self.A - S6)
        dS4 = self.k3 * S3 * (self.A - S6) - self.k4 * S4 * S5 - self.kappa *(S4 - S7)
        dS5 = self.k2 * S2 * (self.N - S5) - self.k4 * S4 * S5 - self.k6 * S2 * S5
        dS6 = 2. * self.k3 * S3 * (self.A - S6) - self.k5 * S6
        dS7 = self.psi * self.kappa * (S4 - S7) - self.k * S7
        return (torch.stack([dS1, dS2, dS3, dS4, dS5, dS6, dS7], dim=1).view(-1,1,7) + self.net(y)).to(device)
    
    def make_nn(self, structure):
        '''
        Structure should contain:
        1. Input size
        2. Output size 
        3. Size for each hidden layers list of (10,20,30,40,50) of length equal to num of hidden layers
        '''
        input_dim = structure[0]
        output_dim = structure[1]
        num_layers = len(structure[2])
        hidden_sizes = structure[2]
        modules = []
        
        for i in range(num_layers):
            if i==0:
                modules.append(nn.Linear(input_dim,hidden_sizes[i]))
                #print(f"Layer {i}")
                modules.append(nn.Tanh())
            
            elif i<num_layers:
                modules.append(nn.Linear(hidden_sizes[i-1],hidden_sizes[i]))
                #print(f"Layer {i}")
                modules.append(nn.Tanh())
            
            else:
                pass

        modules.append(nn.Linear(hidden_sizes[-1],output_dim))
        return nn.Sequential(*modules)
