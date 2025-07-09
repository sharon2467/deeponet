import numpy as np
from scipy.special import jv,lpmv
import scipy.special as sc
class Data_Generator():
    def __init__(self,config):
        self.l=config['length']/2
        self.n=config['N']
        #order是指使用laplace方程解的阶数，只有在cubic,cylinder,sphirical三个模式下有效
        self.order=config['order']
        self.sample=config['sample']
        #以下是几何参数，只在circle,rectangle,reccirc模式下有效
        #circle模式下radius1是圆的半径,dz是两圆的距离，rectangle模式下a是x轴长度，b是y轴长度，dz是两圆的距离。其他几何参数没有使用
        #reccirc模式下所有几何参数均有使用，三个电流也有使用，与pinn相同
        self.dx=config['dx']
        self.dy=config['dy']
        self.dz=config['dz']
        self.radius1=config['radius1']
        self.radius2=config['radius2']
        self.a=config['a']
        self.b=config['b']
        self.Ix=config['Ix']
        self.Iy=config['Iy']
        self.Iz=config['Iz']
        #rotation_list是指在field_manager中使用的旋转方式
        #rotation_on是指是否使用旋转方式
        if(config['rotation_on']):
            self.rotation_list=['I','x','y','z']
        else:
            self.rotation_list=['I']
        #为了省几个参数，order_on和phase_on在选择前三种和后三种磁场配置时有不同的作用
        #order_on在使用前三种磁场配置时是指是否使用不同的阶数，若开启则磁场配置会包含从0-order的阶数，若关闭则只会有order阶的磁场
        #在使用后三种磁场配置时，是指几何参数的波动范围，设为0则磁场配置全部使用一样的几何参数，不为0则所有几何参数都在正负order_on的范围内均匀取值
        if(config['order_on'] and (not config['order_on']%1)):
            self.order_list=np.arange(config['order'])
        else:
            self.order_list=np.array([1])*config['order']
        self.order_on=config['order_on']
        #phase类似于order，在laplace方程的解中允许包含相位参数，在使用前三种磁场配置时若开启则磁场配置会包含从0到2*pi的相位，若关闭则只会有0相位的磁场
        #在使用reccirc磁场配置时是指电流的波动范围，设为0则磁场配置全部使用相同电流，不为0则所有电流都在正负phase_on的范围内均匀取值
        self.phase_on=config['phase_on']  
        self.rotation_on=config['rotation_on']
        #共有几组磁场配置
        self.sets=config['sets']
    def pos_generation(self):
        x=np.linspace(-self.l,self.l,self.n)
        y=np.linspace(-self.l,self.l,self.n)
        X,Y=np.meshgrid(x,y)
        X=np.ravel(X)
        Y=np.ravel(Y)
        Z=np.ones(np.size(X))*self.l
        pos=np.transpose(np.array([X,Y]))
        #通过insert函数生成六个表面上的探测器坐标，代码简短快速，和pinn内的显式设定等价
        for i in range(6):
            if(i==0):
                pos1=np.insert(pos,i//2,Z*(-1)**(i),axis=1)
            else:
                pos1=np.concatenate((pos1,np.insert(pos,i//2,Z*(-1)**(i),axis=1)),axis=0)
        return pos1
    #以下三个解的内容均来自于laplace方程的解，分别对应于cubic,cylinder,spherical三种模式，具体自行参考数学物理方程
    # 每个解都有ij两个阶数和phase三个相位参数，阶数正如角量子数和磁量子数一样。通常j<i。
    def cubic_field(self,pos,i,j,phase):
        Bx=i*np.pi*np.cos(i*pos[:,0]*np.pi+phase[0])*np.sin(j*pos[:,1]*np.pi+phase[1])*np.sinh(np.sqrt(i**2+j**2)*pos[:,2]*np.pi+phase[2])
        By=j*np.pi*np.sin(i*pos[:,0]*np.pi+phase[0])*np.cos(j*pos[:,1]*np.pi+phase[1])*np.sinh(np.sqrt(i**2+j**2)*pos[:,2]*np.pi+phase[2])
        Bz=np.sqrt(i**2+j**2)*np.pi*np.sin(i*pos[:,0]*np.pi+phase[0])*np.sin(j*pos[:,1]*np.pi+phase[1])*np.cosh(np.sqrt(i**2+j**2)*pos[:,2]*np.pi+phase[2])
        B=np.transpose(np.array([Bx,By,Bz]))
        return B
    def cylinder_field(self,pos,i,j,phase):
        r=np.sqrt(pos[:,0]**2+pos[:,1]**2)
        theta=np.arctan2(pos[:,1],pos[:,0])
        Br=j*0.5*((jv(i-1,j*r)-jv(i+1,j*r)))*np.sin(i*theta+phase[1])*np.sinh(j*pos[:,2]+phase[2])
        Btheta=i*jv(i,j*r)*np.cos(i*theta+phase[1])*np.sinh(j*pos[:,2]+phase[2])/r
        Bz=j*jv(i,j*r)*np.sin(i*theta+phase[1])*np.cosh(j*pos[:,2]+phase[2])
        Bx=Br*np.cos(theta)-Btheta*np.sin(theta)
        By=Br*np.sin(theta)+Btheta*np.cos(theta)
        B=np.transpose(np.array([Bx,By,Bz]))
        return B
    def spherical_field(self,pos,i,j,phase):
        r=np.sqrt(pos[:,0]**2+pos[:,1]**2+pos[:,2]**2)
        theta=np.arctan2(np.sqrt(pos[:,0]**2+pos[:,1]**2),pos[:,2])
        phi=np.arctan2(pos[:,1],pos[:,0])
        Br=(i*r**(i-1))*lpmv(j,i,np.cos(theta))*np.sin(j*phi+phase[1])
        Btheta=(r**i)*(np.cos(theta)*j/np.sin(theta)*lpmv(j,i,np.cos(theta))+lpmv(j+1,i,np.cos(theta)))*np.sin(j*phi+phase[1])/r
        Bphi=(r**i)*lpmv(j,i,np.cos(theta))*np.cos(j*phi+phase[1])*j/r/np.sin(theta)
        Bx=Br*np.sin(theta)*np.cos(phi)+Btheta*np.cos(theta)*np.cos(phi)-Bphi*np.sin(phi)
        By=Br*np.sin(theta)*np.sin(phi)+Btheta*np.cos(theta)*np.sin(phi)+Bphi*np.cos(phi)
        Bz=Br*np.cos(theta)-Btheta*np.sin(theta)
        B=np.transpose(np.array([Bx,By,Bz]))
        return B
    #以下内容与pinn基本相同，同样定义了可以任意旋转的磁场
    def circB_xy(self,pos,radius):
        x_prime=pos[:,0]
        y_prime=pos[:,1]
        z_prime=pos[:,2]
        #Defining some parameters to be used in the formulas
         # 定义一些参数，用于公式中
        r_sq_prime = x_prime**2 + y_prime**2 + z_prime**2
        rho_sq_prime = x_prime**2 + y_prime**2
        alpha_sq_prime = radius**2 + r_sq_prime - 2. * np.sqrt(rho_sq_prime)*radius
        beta_sq_prime = radius**2 + r_sq_prime + 2. * np.sqrt(rho_sq_prime)*radius
        k_sq_prime = 1. - alpha_sq_prime / beta_sq_prime

        # 计算椭圆积分
        e_k_sq_prime = sc.ellipe(k_sq_prime)
        k_k_sq_prime = sc.ellipk(k_sq_prime)

        # 计算旋转后的磁场分量
        Bx_prime = x_prime * z_prime / (2*alpha_sq_prime * rho_sq_prime * np.sqrt(beta_sq_prime)) * ((radius**2 + r_sq_prime) * e_k_sq_prime - alpha_sq_prime * k_k_sq_prime)
        By_prime = y_prime * Bx_prime / x_prime
        Bz_prime = 1 / (2*alpha_sq_prime * np.sqrt(beta_sq_prime)) * ((radius**2 - r_sq_prime) * e_k_sq_prime + alpha_sq_prime * k_k_sq_prime)

        #return np.concatenate([Bx.reshape(-1,1),By.reshape(-1,1),Bz.reshape(-1,1)], axis=1)*-100
        return np.transpose(np.array([Bx_prime,By_prime,Bz_prime]))
    def circB_rotate(self,pos,theta,phi,radius):
        x=pos[:,0]
        y=pos[:,1]
        z=pos[:,2]
        #theta是绕y轴正向旋转的角度，phi是绕z轴正向旋转的角度
        pos[:,0]=x*np.cos(theta)*np.cos(phi)+y*np.cos(theta)*np.sin(phi)-z*np.sin(theta)
        pos[:,1]=-x*np.sin(phi)+y*np.cos(phi)
        pos[:,2]=x*np.sin(theta)*np.cos(phi)+y*np.sin(theta)*np.sin(phi)+z*np.cos(theta)
        B_prime=self.circB_xy(pos,radius)
        Bx=B_prime[:,0]*np.cos(theta)*np.cos(phi)-B_prime[:,1]*np.sin(phi)+B_prime[:,2]*np.sin(theta)*np.cos(phi)
        By=B_prime[:,0]*np.cos(theta)*np.sin(phi)+B_prime[:,1]*np.cos(phi)+B_prime[:,2]*np.sin(theta)*np.sin(phi)
        Bz=-B_prime[:,0]*np.sin(theta)+B_prime[:,2]*np.cos(theta)
        return np.transpose(np.array([Bx,By,Bz]))
    def lineB(self,pos,start,end):
        r12=end-start
        r10=pos-start
        r20=pos-end
        cos1=np.sum(r12*r10,1)/(np.linalg.norm(r12)*np.linalg.norm(r10,axis=1))
        cos2=np.sum(r12*r20,1)/(np.linalg.norm(r12)*np.linalg.norm(r20,axis=1))
        r=np.linalg.norm(np.cross(r10,r20),axis=1)/np.linalg.norm(r12)
        B=np.cross(r10,r20)
        B=B*(1/np.linalg.norm(B,axis=1)/r*(cos1-cos2))[:,None]
        return B
    def recB_xy(self,pos,a,b):
        #x轴长为a，y轴长为b
        pos1=np.array([a/2,b/2,0])
        pos2=np.array([-a/2,b/2,0])
        pos3=np.array([-a/2,-b/2,0])
        pos4=np.array([a/2,-b/2,0])
        B=self.lineB(pos,pos1,pos2)+self.lineB(pos,pos2,pos3)+self.lineB(pos,pos3,pos4)+self.lineB(pos,pos4,pos1)
        return B
    def reccircB(self,pos,a,b,radius1,radius2,Ix,Iy,Iz,dx,dy,dz):
        pos1=pos.copy()
        pos1[:,1]=pos[:,1]-dy/2
        field = self.circB_rotate(pos1,np.pi/2,np.pi/2,radius1)*Iy
        pos1[:,1]=pos[:,1]+dy/2
        field += self.circB_rotate(pos1,np.pi/2,np.pi/2,radius1)*Iy
        pos1[:,1]=pos[:,1]
        pos1[:,0]=pos[:,0]-dx/2
        field += self.circB_rotate(pos1,np.pi/2,0,radius2)*Ix
        pos1[:,0]=pos[:,0]+dx/2
        field += self.circB_rotate(pos1,np.pi/2,0,radius2)*Ix
        pos1[:,0]=pos[:,0]
        pos1[:,2]=pos[:,2]-dz/2
        field += self.recB_xy(pos1,a,b)*Iz
        pos1[:,2]=pos[:,2]+dz/2
        field += self.recB_xy(pos1,a,b)*Iz
        return field
    def field_manager(self,pos,i,j,phase,mode,rotation,radius1,radius2,a,b,dx,dy,dz,Ix,Iy,Iz):
        #field_manager将完成上述基础函数的旋转，平移和组合。例如上述函数的都只计算单个在原点的线圈的磁场，这里将两个线圈上下平移后才能构成亥姆霍兹线圈。
        pos1=pos.copy()
        if(rotation=='x'):
            pos1[:,1]=pos[:,2]
            pos1[:,2]=-pos[:,1]
        elif(rotation=='y'):
            pos1[:,2]=pos[:,0]
            pos1[:,0]=-pos[:,2]
        elif(rotation=='z'):
            pos1[:,0]=pos[:,1]
            pos1[:,1]=-pos[:,0]
        if(mode=='cubic'):
            B=self.cubic_field(pos1,i,j,phase)
        elif(mode=='spherical'):
            B=self.spherical_field(pos1,i,j,phase)
        elif(mode=='cylinder'):
            B=self.cylinder_field(pos1,i,j,phase)
        elif(mode=='circle'):
            pos2=pos1.copy()
            pos2[:,2]=pos1[:,2]-dz/2
            B=self.circB_xy(pos2,radius1)
            pos2[:,2]=pos1[:,2]+dz/2
            B=B+self.circB_xy(pos2,radius1)
        elif(mode=='rectangle'):
            pos2=pos1.copy()
            pos2[:,2]=pos1[:,2]-dz/2
            B=self.recB_xy(pos2,a,b)
            pos2[:,2]=pos1[:,2]+dz/2
            B=B+self.recB_xy(pos2,a,b)
        elif(mode=='reccirc'):
            B=self.reccircB(pos1,a,b,radius1,radius2,dx,dy,dz,Ix,Iy,Iz)
        else:
            raise ValueError('mode not supported')
        B1=B.copy()
        if(rotation=='x'):
            B1[:,2]=B[:,1]
            B1[:,1]=-B[:,2]
        elif(rotation=='y'):
            B1[:,2]=-B[:,0]
            B1[:,0]=B[:,2]
        elif(rotation=='z'):
            B1[:,1]=B[:,0]
            B1[:,0]=-B[:,1]
        return B1
    def data_generation(self,pos,mode):
        data=np.zeros((self.sets,len(pos)*3))
        y_data=np.zeros((self.sample,3,self.sets))
        y_B_data=np.zeros((self.sample,3,self.sets))
        for i in range(self.sets):
            phase=np.random.uniform(-self.phase_on,self.phase_on,3)
            a = np.random.uniform(self.a - self.order_on, self.a + self.order_on)
            b = np.random.uniform(self.b - self.order_on, self.b + self.order_on)
            dx = np.random.uniform(self.dx - self.order_on, self.dx + self.order_on)
            dy = np.random.uniform(self.dy - self.order_on, self.dy + self.order_on)
            dz = np.random.uniform(self.dz - self.order_on, self.dz + self.order_on)
            radius1 = np.random.uniform(self.radius1 - self.order_on, self.radius1 + self.order_on)
            radius2 = np.random.uniform(self.radius2 - self.order_on, self.radius2 + self.order_on)
            Ix=np.random.uniform(self.Ix - self.phase_on, self.Ix + self.phase_on)
            Iy=np.random.uniform(self.Iy - self.phase_on, self.Iy + self.phase_on)
            Iz=np.random.uniform(self.Iz - self.phase_on, self.Iz + self.phase_on)
            y=np.random.uniform(-self.l,self.l,(self.sample,3))
            #随机选择旋转方式和阶数
            rotation=np.random.choice(self.rotation_list)
            random_order_i = np.random.choice(self.order_list)
            random_order_j = np.random.choice(self.order_list)   
            #调用field_manager计算磁场，注意传入的参数不会被全部使用
            data[i,:]=np.ravel(self.field_manager(pos, random_order_i, random_order_j, phase, mode, rotation, radius1, radius2, a, b, dx, dy, dz, Ix, Iy, Iz))
            y_data[:,:,i]=y
            y_B_data[:,:,i]=self.field_manager(y,random_order_i,random_order_j,phase,mode,rotation,radius1,radius2,a,b,dx,dy,dz,Ix,Iy,Iz)
        return data,y_data,y_B_data
    def meshgrid_generation(self,mode):
        # 这个函数生成用于conv模型的网格数据，顺序为(height, width,channel,batch)
        # 不推荐这样的实践，推荐的顺序为(batch, channel,height,width)
        # 因为线性索引变化最快的维度是最后一个维度，使用推荐的顺序可以确保在reshape后不同batch的数据不会混淆
        # 后续有意愿的话可进行修改
        x=np.linspace(-self.l,self.l,self.n)
        y=np.linspace(-self.l,self.l,self.n)
        data=np.zeros((self.n,self.n,18,self.order**2*4))
        y_data=np.zeros((self.sample,3,self.order**2*4))
        y_B_data=np.zeros((self.sample,3,self.order**2*4))
        X,Y=np.meshgrid(x,y)
        X=np.ravel(X)
        Y=np.ravel(Y)
        Z=np.ones(np.size(X))*self.l
        pos=np.transpose(np.array([X,Y]))
        for i1 in range(self.order):
            for j1 in range(self.order):
                if(self.phase_on):
                    phase=np.random.uniform(-np.pi,np.pi,3)
                else:
                    phase=[0,0,0]
                y=np.random.uniform(-self.l,self.l,(self.sample,3))
                r=-1
                for rotation in self.rotation_list:
                    r=r+1
                    y_data[:,:,self.order*4*i1+4*j1+r]=y
                    y_B_data[:,:,self.order*4*i1+4*j1+r]=self.field_manager(y,self.order_list[i1],self.order_list[j1],phase,mode,rotation)
                    for i in range(6):
                        pos1=np.insert(pos,i//2,Z*(-1)**(i),axis=1)
                        if(i==0):
                            probe_pos=pos1
                        else:
                            probe_pos=np.concatenate((probe_pos,pos1),axis=0)
                        temp=self.field_manager(pos1,self.order_list[i1],self.order_list[j1],phase,mode,rotation)
                        for j in range(3):
                            data[:,:,3*i+j,self.order*4*i1+4*j1+r]=np.reshape(temp[:,j],(self.n,self.n))
        return data,y_data,y_B_data,probe_pos


                
