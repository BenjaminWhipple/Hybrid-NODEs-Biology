import torch
import torch.nn as nn

device = torch.device('cpu')

class hybridODE(nn.Module):

    def __init__(self, p0,structure):
        super(hybridODE, self).__init__()

        self.paramsODE = p0

        self.beta = self.paramsODE[0]
        self.gamma = self.paramsODE[1]
        self.delta = self.paramsODE[2]

        self.net = self.make_nn(structure)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        S1 = y.view(-1,4)[:,0]
        S2 = y.view(-1,4)[:,1]
        T = y.view(-1,4)[:,2]
        A = y.view(-1,4)[:,3]
        
        dS1 = self.beta*S1*A - self.gamma*S1*S2
        dS2 = self.gamma*S1*S2 - self.delta*S2

        shape_dS1 = dS1.shape

        dT = torch.tensor([1.0]).expand(shape_dS1[0])
        dA = 0.0*A

        # These following 3 lines maintain compatibility with the torchdiffeq interface while forcing the network to only apply to the A compartment.
        shape = self.net(y).shape
        filler = torch.full((shape[0],shape[1],3),fill_value=0)
        net_out = torch.cat((filler,self.net(y)),dim=2)

        return (torch.stack([dS1, dS2,dT,dA], dim=1).view(-1,1,4) + self.net(y)).to(device)
    
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
