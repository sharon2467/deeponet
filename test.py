import numpy as np  
import matplotlib.pyplot as plt  
from mpl_toolkits.mplot3d import Axes3D  
from data import *

# 创建一个新的图形  
fig = plt.figure()  
#ax = fig.add_subplot(111, projection='3d')  
config = {}
config['length']=0.5
config['order']=2
config['sample']=1
config['radius1']=1
config['radius2']=1
config['a']=1
config['b']=1
config['dx']=4
config['dy']=4
config['dz']=4
config['N']=10
config['order']=2
config['rotation_on']=1
config['phase_on']=1
config['order_on']=1
config['Ix']=1
config['Iy']=1
config['Iz']=1
field = Data_Generator(config)

N_val=100
L=field.l/2
y_test_np_grid = np.linspace(-L, L, N_val)
x_test_np_grid = np.linspace(-L, L, N_val)
z_test_np_grid = np.linspace(-L, L, N_val)
xx, yy, zz = np.meshgrid(x_test_np_grid, y_test_np_grid, z_test_np_grid, sparse=False, indexing='ij')
temp_final = np.zeros(np.append(np.shape(xx),3))
xxravel = xx.ravel()
yyravel = yy.ravel()
zzravel = zz.ravel()
i=3
j=1
temp_final = field.field_manager(np.transpose(np.array([xxravel,yyravel,zzravel])),i,j,[1,1,1],'circle','I',1,1,2,2,2,2,2)
pos=np.transpose(np.array([xxravel,yyravel,zzravel]))
phase=[0,0,0]
temp_final = np.reshape(temp_final,(N_val,N_val,N_val,3))
#print(np.max(np.abs(temp_final)))
# 通过画箭头来表示矢量  
#for i in range(xx.size):  
    #idx=np.unravel_index(i, np.shape(xx))
    #ax.quiver(xxravel[i],yyravel[i],zzravel[i], *temp_final[idx]*0.5)  
grad_x = np.gradient(temp_final[..., 0],field.l/N_val,axis=0)
grad_y = np.gradient(temp_final[..., 1],field.l/N_val,axis=1)
grad_z = np.gradient(temp_final[..., 2],field.l/N_val,axis=2)
#dBx=-(i*np.pi)**2*np.sin(i*pos[:,0]*np.pi+phase[0])*np.sin(j*pos[:,1]*np.pi+phase[1])*np.sinh(np.sqrt(i**2+j**2)*pos[:,2]*np.pi+phase[2])
#dBy=-(j*np.pi)**2*np.sin(i*pos[:,0]*np.pi+phase[0])*np.sin(j*pos[:,1]*np.pi+phase[1])*np.sinh(np.sqrt(i**2+j**2)*pos[:,2]*np.pi+phase[2])
#dBz=(np.sqrt(i**2+j**2)*np.pi)**2*np.sin(i*pos[:,0]*np.pi+phase[0])*np.sin(j*pos[:,1]*np.pi+phase[1])*np.sinh(np.sqrt(i**2+j**2)*pos[:,2]*np.pi+phase[2])
#dBx = np.reshape(dBx,(N_val,N_val,N_val))
#dBy = np.reshape(dBy,(N_val,N_val,N_val))
#dBz = np.reshape(dBz,(N_val,N_val,N_val))
#print(np.mean(np.abs(grad_y-dBy)))
curl_x = np.gradient(temp_final[...,2], axis=1) - np.gradient(temp_final[...,1], axis=2)
curl_y = np.gradient(temp_final[...,0], axis=2) - np.gradient(temp_final[...,2], axis=0)
curl_z = np.gradient(temp_final[...,1], axis=0) - np.gradient(temp_final[...,0], axis=1)
print(np.mean(np.abs(grad_x+grad_y+grad_z)))
print(np.mean(np.sqrt(curl_x**2+curl_y**2+curl_z**2)))
print(np.mean(np.sqrt(np.sum(temp_final**2,axis=3))))
# 设置坐标轴标签  
#ax.set_xlabel('X')  
#ax.set_ylabel('Y')  
# 显示图形  
#plt.show()
CS = plt.contourf(xx[:,:,0],yy[:,:,0],temp_final[:,:,5,2], cmap='jet')
plt.colorbar(CS)
plt.show()