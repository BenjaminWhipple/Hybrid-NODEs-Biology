import torch
import torch.nn as nn

device = torch.device('cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu')

class Known_Params_HybridODE(nn.Module):

    def __init__(self,structure):
        super(Known_Params_HybridODE, self).__init__()
        

        self.beta1 = 0.3
        self.beta2 = 0.2
        self.gamma1 = 0.1
        self.gamma2 = 0.1
        
        """
        # Testing incorrect parameters
        self.beta1 = 0.2
        self.beta2 = 0.1
        self.gamma1 = 0.2
        self.gamma2 = 0.2
        """

        self.net = self.make_nn(structure)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        S1 = y.view(-1,6)[:,0]
        I1 = y.view(-1,6)[:,1]
        R1 = y.view(-1,6)[:,2]
        S2 = y.view(-1,6)[:,3]
        I2 = y.view(-1,6)[:,4]
        R2 = y.view(-1,6)[:,5]
        
        N1 = S1 + I1 + R1  # Total for site 1 (not necessarily needed but useful for clarity)
        N2 = S2 + I2 + R2  # Total for site 2
        
        dS1 = -self.beta1 * S1 * I1 / N1
        dI1 = self.beta1 * S1 * I1 / N1 - self.gamma1 * I1
        dR1 = self.gamma1 * I1
        dS2 = -self.beta2 * S2 * I2 / N2
        dI2 = self.beta2 * S2 * I2 / N2 - self.gamma2 * I2
        dR2 = self.gamma2 * I2
        
        return (torch.stack([dS1, dI1, dR1, dS2, dI2, dR2], dim=1).view(-1,1,6) + self.net(y)).to(device)
    
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

