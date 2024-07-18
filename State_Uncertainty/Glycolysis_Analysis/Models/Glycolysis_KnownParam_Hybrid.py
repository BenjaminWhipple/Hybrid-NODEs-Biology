import torch
import torch.nn as nn

device = torch.device('cpu')

class Known_Params_HybridODE(nn.Module):

    def __init__(self, p0, structure):
        super(Known_Params_HybridODE, self).__init__()

        self.paramsODE = p0#nn.Parameter(p0)
        self.J0 = self.paramsODE[0]     #mM min-1
        self.k1 = self.paramsODE[1]     #mM-1 min-1
        self.k2 = self.paramsODE[2]     #mM min-1
        self.k3 = self.paramsODE[3]     #mM min-1
        self.k4 = self.paramsODE[4]     #mM min-1
        self.k5 = self.paramsODE[5]     #mM min-1
        self.k6 = self.paramsODE[6]     #mM min-1
        self.k = self.paramsODE[7]      #min-1
        self.kappa = self.paramsODE[8]  #min-1
        self.q = self.paramsODE[9]
        self.K1 = self.paramsODE[10]    #mM
        self.psi = self.paramsODE[11]
        self.N = self.paramsODE[12]     #mM
        self.A = self.paramsODE[13]     #mM

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
        S7 = y.view(-1,7)[:,5]
        # This reordering provides allows us to concat in the dummy column easier during training.
        S6 = y.view(-1,7)[:,6]


        dS1 = self.J0 - (self.k1*S1*S6)/(1 + (S6/self.K1)**self.q)
        dS2 = 2. * (self.k1*S1*S6)/(1 + (S6/self.K1)**self.q) - self.k2 * S2 * (self.N - S5) - self.k6 * S2 * S5
        dS3 = self.k2 * S2 * (self.N - S5) - self.k3 * S3 * (self.A - S6)
        dS4 = self.k3 * S3 * (self.A - S6) - self.k4 * S4 * S5 - self.kappa *(S4 - S7)
        dS5 = self.k2 * S2 * (self.N - S5) - self.k4 * S4 * S5 - self.k6 * S2 * S5
        # We have a position of ignorance on S6, so we zero it out
        # The way we write dS6 preserves the tensor type.
        dS6 = 0.0*S6#.view(1)
        dS7 = self.psi * self.kappa * (S4 - S7) - self.k * S7
        
        #print(self.net(y).shape)
        # Filler to maintain compatibility.
        shape = self.net(y).shape
        filler = torch.full((shape[0],shape[1],6),fill_value=0)
        net_out = torch.cat((filler,self.net(y)),dim=2)
        #print(torch.stack([dS1, dS2, dS3, dS4, dS5, dS7, dS6], dim=1).view(-1,1,7).shape)
        #return (torch.stack([dS1, dS2, dS3, dS4, dS5, dS7, dS6], dim=1).view(-1,1,7) + self.net(y)).to(device)
        return (torch.stack([dS1, dS2, dS3, dS4, dS5, dS7, dS6], dim=1).view(-1,1,7) + net_out).to(device)
        """
        S1 = y.view(-1,6)[:,0]
        S2 = y.view(-1,6)[:,1]
        S3 = y.view(-1,6)[:,2]
        S4 = y.view(-1,6)[:,3]
        S5 = y.view(-1,6)[:,4]
        #S6 = y.view(-1,7)[:,5]
        S7 = y.view(-1,6)[:,5]
        
        NN = self.net(y)
        print(f"In forward, NN size = {NN.shape}")
        
        print(f"In forward, NN = {NN[-1,0,:]}")
        
        # We just zero out any S6 term.
        dS1 = self.J0 # + NN[0]
        dS2 = - self.k2 * S2 * (self.N - S5) - self.k6 * S2 * S5 #+ NN[1]
        dS3 = self.k2 * S2 * (self.N - S5) - self.k3 * S3 * self.A #+ NN[2]
        dS4 = self.k3 * S3 * self.A - self.k4 * S4 * S5 - self.kappa *(S4 - S7) #+ NN[3]
        dS5 = self.k2 * S2 * (self.N - S5) - self.k4 * S4 * S5 - self.k6 * S2 * S5
        #dS6 = -2. * (self.k1 * S1 * S6) / (1 + (S6 / self.K1)**self.q) + 2. * self.k3 * S3 * (self.A - S6) - self.k5 * S6
        dS7 = self.psi * self.kappa * (S4 - S7) - self.k * S7
        print(f"dS1 = {dS1}")
        print(f"shape = {dS1.shape}")
        
        return (torch.stack([dS1, dS2, dS3, dS4, dS5, dS7], dim=1).view(-1,1,6).to(device))
        """
        
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
