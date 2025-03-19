import torch
from torch import nn
from torch.autograd import grad
import numpy as np
def gradients(u, x):
    return grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True,  only_inputs=True, allow_unused=True)[0]
class DeepONet_Linear(nn.Module):
    def __init__(self, units, num_mid,probes_pos):
        super(DeepONet_Linear, self).__init__()
        num_probes=len(probes_pos[:,0])
        self.branch = nn.Sequential(
            nn.Linear(num_probes * 3, units),
            nn.ReLU(),
            nn.Linear(units, units),
            nn.ReLU(),
            nn.Linear(units, units),
            nn.ReLU(),
            nn.Linear(units, units),
            nn.ReLU(),
            nn.Linear(units, units),
            nn.ReLU(),
            nn.Linear(units, num_mid)
        )
        self.trunk = nn.Sequential(
            nn.Linear(3, units),
            nn.ReLU(),
            nn.Linear(units, units),
            nn.ReLU(),
            nn.Linear(units, units),
            nn.ReLU(),
            nn.Linear(units, units),
            nn.ReLU(),
            nn.Linear(units, units),
            nn.ReLU(),
            nn.Linear(units, num_mid)
        )
        self.num_probes = num_probes
        self.num_mid = num_mid
        self.probes_pos=probes_pos
        # 定义一个线性层，将两个 num_mid 长度的向量混合成一个长度为 3 的向量
        self.mix_layer = nn.Sequential(
            nn.Linear(num_mid, units),
            nn.ReLU(),
            nn.Linear(units, 3)
        )

    def forward(self, branch_input, trunk_input):
        #print(branch_input.shape)
        #print(trunk_input.shape)
        branch_output = self.branch(branch_input)
        trunk_output = self.trunk(trunk_input)
        #print(branch_output.shape)
        #print(trunk_output.shape)
        # 将 branch_output 和 trunk_output 拼接在一起
        #combined_output = torch.cat((branch_output.repeat(trunk_output.size()[0],1), trunk_output), dim=1)
        if(len(branch_output.shape)<=1):
            combined_output = torch.mul(branch_output.unsqueeze(0), trunk_output)
        else:
            combined_output = torch.mul(branch_output, trunk_output)
        #print(combined_output.shape)
        # 通过线性层混合成一个长度为 3 的向量
        output = self.mix_layer(combined_output)
        #print(output.shape)
        return output
class DeepONet_Conv(nn.Module):
    def __init__(self, units, num_mid,n,probes_pos):
        self.probes_pos=probes_pos
        super(DeepONet_Conv, self).__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(18, units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(units, units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(units, units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(units, units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(units, units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(units, 1, kernel_size=3, padding=1)
        )
        self.trunk = nn.Sequential(
            nn.Linear(3, units),
            nn.ReLU(),
            nn.Linear(units, units),
            nn.ReLU(),
            nn.Linear(units, units),
            nn.ReLU(),
            nn.Linear(units, units),
            nn.ReLU(),
            nn.Linear(units, units),
            nn.ReLU(),
            nn.Linear(units, units),
            nn.ReLU(),
            nn.Linear(units, units),
            nn.ReLU(),
            nn.Linear(units, num_mid)
        )
        self.num_mid = num_mid
        self.n=n

            # 定义一个线性层，将两个 num_mid 长度的向量混合成一个长度为 3 的向量
        self.mixlayer = nn.Sequential(
            nn.Linear(n**2+num_mid, units),
            nn.ReLU(),
            nn.Linear(units, units),
            nn.ReLU(),
            nn.Linear(units, units),
            nn.ReLU(),
            nn.Linear(units, units),
            nn.ReLU(),
            nn.Linear(units, 3)
        )

    def forward(self,branch_input,trunk_input):
        #[Bx_mean,By_mean,Bz_mean]=[torch.mean(branch_input[:,:,::3]),torch.mean(branch_input[:,:,1::3]),torch.mean(branch_input[:,:,2::3])]
        #[Bx_std,By_std,Bz_std]=[torch.std(branch_input[:,:,::3]),torch.std(branch_input[:,:,1::3]),torch.std(branch_input[:,:,2::3])]
        #branch_input1=branch_input.clone()
        #branch_input1[:,:,::3]=(branch_input[:,:,::3]-Bx_mean)/Bx_std
        #branch_input1[:,:,1::3]=(branch_input[:,:,1::3]-By_mean)/By_std
        #branch_input1[:,:,2::3]=(branch_input[:,:,2::3]-Bz_mean)/Bz_std
        branch_output = self.branch(branch_input.permute(2,0,1).unsqueeze(0))
        trunk_output = self.trunk(trunk_input)
        # 将 branch_output 和 trunk_output 拼接在一起
        combined_output = torch.cat((torch.ravel(branch_output).repeat(trunk_output.size()[0],1), trunk_output), dim=1)
        # 通过线性层混合成一个长度为 3 的向量
        output = self.mixlayer(combined_output)
        #output1=output.clone()
        #output1[:,0]=(output[:,0]*Bx_std+Bx_mean)
        #output1[:,1]=(output[:,1]*By_std+By_mean)
        #output1[:,2]=(output[:,2]*Bz_std+Bz_mean)
        return output
class PINN_Loss(nn.Module):
    #初始化神经网络输入，定义输入参数
    def __init__(self, N_f, L, device, addBC,Lambda,pinn):
        super(PINN_Loss, self).__init__() #继承tf.keras.Model的功能
        self.N_f = N_f
        self.L = L
        self.device = device
        if(addBC==0):
            self.addBC = False
        if(addBC==1):
            self.addBC = True
        self.Lambda=Lambda*pinn
        self.pinn=pinn
    def loss(self, data,Bdata, labels, model):
        device = self.device            
        train_x = data[:,0].reshape(-1,1).requires_grad_(True)
        train_y = data[:,1].reshape(-1,1).requires_grad_(True)
        train_z = data[:,2].reshape(-1,1).requires_grad_(True)
        B = model(Bdata,torch.cat((train_x, train_y, train_z), axis=1).to(torch.float32))            
        if(self.pinn):

            B_x = B[:,0].requires_grad_(True)
            B_y = B[:,1].requires_grad_(True)
            B_z = B[:,2].requires_grad_(True)
            dx = gradients(B_x, train_x)
            dy = gradients(B_y, train_y)
            dz = gradients(B_z, train_z)
            dzy = gradients(B_z, train_y)
            dzx = gradients(B_z, train_x)
            dyz = gradients(B_y, train_z)
            dyx = gradients(B_y, train_x)
            dxy = gradients(B_x, train_y)
            dxz = gradients(B_x, train_z)        
            loss_BC_div = torch.mean(torch.square(dx+dy+dz))
            loss_BC_cul = torch.mean(torch.square(dzy - dyz) + torch.square(dxz - dzx) + torch.square(dyx - dxy))
            y_f = np.random.default_rng().uniform(low = -self.L/2, high = self.L/2, size = ((self.N_f, 1)))
            x_f = np.random.default_rng().uniform(low = -self.L/2, high = self.L/2, size = ((self.N_f, 1)))
            z_f = np.random.default_rng().uniform(low = -self.L/2, high = self.L/2, size = ((self.N_f, 1)))
            self.x_f = torch.tensor(x_f, dtype = torch.float32).to(device).requires_grad_(True)
            self.y_f = torch.tensor(y_f, dtype = torch.float32).to(device).requires_grad_(True)
            self.z_f = torch.tensor(z_f, dtype = torch.float32).to(device).requires_grad_(True)
            temp_pred = model(Bdata,torch.cat((self.x_f, self.y_f, self.z_f), axis=1))
            temp_ux = temp_pred[:,0].requires_grad_(True)
            temp_uy = temp_pred[:,1].requires_grad_(True)
            temp_uz = temp_pred[:,2].requires_grad_(True)
            u_x = gradients(temp_ux, self.x_f)
            u_y = gradients(temp_uy, self.y_f)
            u_z = gradients(temp_uz, self.z_f)
            u_zy = gradients(temp_uz, self.y_f) #dBz_f/dy_f
            u_zx = gradients(temp_uz, self.x_f) #dBz_f/dx_f
            u_yz = gradients(temp_uy, self.z_f) #dBy_f/dz_f
            u_yx = gradients(temp_uy, self.x_f) #dBy_f/dx_f
            u_xz = gradients(temp_ux, self.z_f) #dBx_f/dz_f
            u_xy = gradients(temp_ux, self.y_f) #dBx_f/dy_f
            #计算散度：div B = ∇·B = dBx_f/dx_f + dBy_f/dy_f + dBz_f/dz_f
            loss_f = torch.mean(torch.square(u_x + u_y + u_z))
            #计算散度的平方作为loss_∇·B
            #计算旋度的模方：|∇×B|^2，作为loss_∇×B
            loss_cross = torch.mean(torch.square(u_zy - u_yz) + torch.square(u_xz - u_zx) + torch.square(u_yx - u_xy))
        else:
        #计算采样磁场大小和预测磁场大小的差，作为loss_B
            loss_f=torch.tensor(1)
            loss_cross=torch.tensor(1)
            loss_BC_div=torch.tensor(1)
            loss_BC_cul=torch.tensor(1)
        loss_u = torch.mean(torch.square(B - labels))
        if(self.addBC):
            loss = loss_f*self.Lambda + loss_u + loss_cross*self.Lambda + loss_BC_div*self.Lambda+ loss_BC_cul*self.Lambda
        else:
            loss  = loss_f*self.Lambda + loss_u + loss_cross*self.Lambda
        return loss_f, loss_u, loss_cross, loss_BC_div, loss_BC_cul, loss