import torch
from torch import nn
from torch.autograd import grad
import numpy as np
#deeponet分离了需要得知磁场点的位置信息的处理和探测器处磁场信息的处理，分别交给了trunk和branch网络。中间由middlelayer连接。
def gradients(u, x):
    return grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True,  only_inputs=True, allow_unused=True)[0]
class DeepONet_Linear(nn.Module):
    def __init__(self, units, num_mid,probes_pos):
        super(DeepONet_Linear, self).__init__()
        num_probes=len(probes_pos[:,0])
        #branch网络的输入是二维数组，每行代表一个磁场配置下所有探测器的数据，三个维度上num_probes个探测器的数据被展平成一维向量
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
        branch_output = self.branch(branch_input)
        trunk_output = self.trunk(trunk_input)
        # 将 branch_output 和 trunk_output 拼接在一起
        #combined_output = torch.cat((branch_output.repeat(trunk_output.size()[0],1), trunk_output), dim=1)
        #为了避免一维数组不能与二维相乘，为其添加维度
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
        #branch网络输入可以是四维数组(batch,channel,height,width)，3个方向6个面共18个通道，每个通道都是一个面均匀分布的探测器上某个方向的磁场
        #但实际上因为函数train_data_generation的处理，输入是一个三维数组，第一二维是点位数目，第三维是每个点位在三个方向六个面上的磁场数据，即(height,width,channel)
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
        #由于没有采用(batch,channel,height,width)的顺序，所以需要对branch_input进行转置和增加维度
        #建议日后的编程实践改为采用这种顺序
        branch_output = self.branch(branch_input.permute(2,0,1).unsqueeze(0))
        trunk_output = self.trunk(trunk_input)
        # 将 branch_output 和 trunk_output 拼接在一起
        combined_output = torch.cat((torch.ravel(branch_output).repeat(trunk_output.size()[0],1), trunk_output), dim=1)
        # 通过线性层混合成一个长度为 3 的向量
        output = self.mixlayer(combined_output)
        return output
class PINN_Loss(nn.Module):
    #初始化神经网络输入，定义输入参数
    def __init__(self, N_f, L, device, addBC,Lambda,pinn):
        super(PINN_Loss, self).__init__()
        #实际上应该是deeponetloss。单纯的deeponetloss没有pinn的部分，但添加了一个开关，可以选择是否添加pinn的部分。
        #N_f是采样点的数量，L是采样范围，device是计算设备，addBC是是否添加边界条件，Lambda是损失函数的权重
        #pinn是一个开关，表示是否使用pinn的部分
        #但出于未知原因，计算pinn损失时会产生错误和梯度爆炸，所以实际上这部分尚待完成
        #另外，pinn与deeponet结合的工作已经有人做过了
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
            #这部分与pinn相同
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