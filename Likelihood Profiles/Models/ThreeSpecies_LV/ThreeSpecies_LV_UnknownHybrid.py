import torch
import torch.nn as nn

device = torch.device('cpu')

class UnknownParam_HNDE(nn.Module):

    def __init__(self, p0,structure):
        super(UnknownParam_HNDE, self).__init__()

        #self.paramsODE = nn.Parameter(p0)
        beta,gamma,delta = p0
        #        J0,k1,k2,k3,k4,k5,k6,k,kappa,q,K1,psi,N,A = p0
        #self.J0 = nn.Parameter(torch.tensor(J0).to(device))

        self.beta = nn.Parameter(torch.tensor(beta).to(device))
        self.gamma = nn.Parameter(torch.tensor(gamma).to(device))
        self.delta = nn.Parameter(torch.tensor(delta).to(device))

        self.net = self.make_nn(structure)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        S1 = y.view(-1,3)[:,0]
        S2 = y.view(-1,3)[:,1]
        #T = y.view(-1,4)[:,2]
        A = y.view(-1,3)[:,2]
        
        dS1 = self.beta*S1*A - self.gamma*S1*S2
        dS2 = self.gamma*S1*S2 - self.delta*S2

        shape_dS1 = dS1.shape

        #dT = torch.tensor([1.0]).expand(shape_dS1[0])
        dA = 0.0*A

        # These following 3 lines maintain compatibility with the torchdiffeq interface while forcing the network to only apply to the A compartment.
        shape = self.net(y).shape
        filler = torch.full((shape[0],shape[1],2),fill_value=0)
        net_out = torch.cat((filler,self.net(y)),dim=2) # Concat to get [0,0, output] to add to final result

        return (torch.stack([dS1, dS2,dA], dim=1).view(-1,1,3) + self.net(y)).to(device)
    
    def make_nn(self, structure):
        '''
        Structure should contain:
        1. Input size
        2. Output size 
        3. Size for each hidden layers list of (10,20,30,40,50) of length equal to num of hidden layers
        Maybe? 3. Activation function for each layer (Tanh)
        '''
        input_dim = structure[0]
        output_dim = structure[1]
        num_layers = len(structure[2])
        hidden_sizes = structure[2]
        modules = []
        

        #print(hidden_sizes)
        for i in range(num_layers):
            print(hidden_sizes[i])
            if i==0:
                #Add input layer
                #print(i)
                modules.append(nn.Linear(input_dim,hidden_sizes[i]))
                modules.append(nn.Tanh())
            
            elif i<num_layers:
                #print(i)
                
                #print(hidden_sizes[i-1])
                #print(hidden_sizes[i])
                modules.append(nn.Linear(hidden_sizes[i-1],hidden_sizes[i]))
                modules.append(nn.Tanh())
            
            else:
                pass
        #print(i)
        modules.append(nn.Linear(hidden_sizes[-1],output_dim))
        
        #print(modules)
        return nn.Sequential(*modules)
