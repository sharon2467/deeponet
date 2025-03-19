import torch
from data import *
import matplotlib.pyplot as plt
import time
def getdB(Bpred, Breal):
    Bx = Bpred[:,0].detach().numpy()
    By = Bpred[:,1].detach().numpy()
    Bz = Bpred[:,2].detach().numpy()
    Bx_r = Breal[:,0]
    By_r = Breal[:,1]
    Bz_r = Breal[:,2]
    dBx = Bx - Bx_r
    dBy = By - By_r
    dBz = Bz - Bz_r
    dB = np.sqrt(Bx**2+By**2+Bz**2) - np.sqrt(Bx_r**2+By_r**2+Bz_r**2)
    return dBx, dBy, dBz, dB
def Eval(model, config, field):
    st=time.time()
    model=model.to('cpu')
    model_type=config['model_type']
    path  = config['path']
    Btype = config['Btype']
    mean  = config['mean_labeldata']
    std   = config['std_labeldata']
    mean_Bdata=config['mean_Bdata']
    std_Bdata=config['std_Bdata']
    mean_posdata=config['mean_posdata']
    std_posdata=config['std_posdata']
    L = config['length']/2
    N_val =10
    y_test_np_grid = np.linspace(-L, L, N_val)
    x_test_np_grid = np.linspace(-L, L, N_val)
    z_test_np_grid = np.linspace(-L, L, N_val)
    xx, yy, zz = np.meshgrid(x_test_np_grid, y_test_np_grid, z_test_np_grid, sparse=False)
    x_test_np = xx.reshape((N_val**3, 1))
    y_test_np = yy.reshape((N_val**3, 1))
    z_test_np = zz.reshape((N_val**3, 1))
    xx_tensor = torch.tensor(x_test_np, dtype=torch.float32)
    yy_tensor = torch.tensor(y_test_np, dtype=torch.float32)
    zz_tensor = torch.tensor(z_test_np, dtype=torch.float32)
    inputs = torch.cat([xx_tensor, yy_tensor, zz_tensor], axis = 1)
    inputs_np = inputs.detach().numpy()
    if(config['Btype']=='reccirc'):
        phase=[1,1,1]
    else:
        phase=[0,0,0]
    temp_final=field.field_manager(inputs_np,field.order,field.order,phase,Btype,'I',config['radius1'],config['radius2'],config['a'],config['b'],config['dx'],config['dy'],config['dz'],config['Ix'],config['Iy'],config['Iz'])
    if(model_type=='DeepONet'):
        temp_final1=field.field_manager(model.probes_pos,field.order,field.order,[0,0,0],Btype,'I',config['radius1'],config['radius2'],config['a'],config['b'],config['dx'],config['dy'],config['dz'],config['Ix'],config['Iy'],config['Iz'])
        B=np.ravel(temp_final1)
        B=(B-mean_Bdata)/std_Bdata
        B=torch.tensor(B,dtype=torch.float32)
        inputs=(inputs-mean_posdata)/std_posdata
        model_output=model(B,inputs)*std+mean
    elif(model_type=='DeepONet_Conv'):
        x=np.linspace(-L,L,model.n)
        y=np.linspace(-L,L,model.n)
        X,Y=np.meshgrid(x,y)
        X=np.ravel(X)
        Y=np.ravel(Y)
        Z=np.ones(np.size(X))*L
        pos=np.transpose(np.array([X,Y]))
        data=np.zeros((model.n,model.n,18))
        for i in range(6):
            pos1=np.insert(pos,i//2,Z*(-1)**(i),axis=1)
            temp=field.field_manager(pos1,field.order,field.order,[0,0,0],Btype,'I')
            for j in range(3):
                data[:,:,3*i+j]=np.reshape(temp[:,j],(model.n,model.n))
        
        data=(data-mean_Bdata)/std_Bdata
        inputs=(inputs-mean_posdata)/std_posdata
        data=torch.tensor(data,dtype=torch.float32)
        model_output=model(data,inputs)*std+mean
    dBx, dBy, dBz, dB = getdB(model_output, temp_final)
    fig_stat = plt.figure(figsize=([16,16]))
    fig_stat.add_subplot(2,2,1)
    plt.hist(dBx, bins=10, label=f"Bx_pred - Bx_real: mean {dBx.mean():.5f} std {dBx.std():.5f}")
    plt.legend()
    plt.yscale('log')
    fig_stat.add_subplot(2,2,2)
    plt.hist(dBy, bins=10, label=f"By_pred - By_real: mean {dBy.mean():.5f} std {dBy.std():.5f}")
    plt.legend()
    plt.yscale('log')
    fig_stat.add_subplot(2,2,3)
    plt.hist(dBz, bins=10, label=f"Bz_pred - Bz_real: mean {dBz.mean():.5f} std {dBz.std():.5f}")
    plt.legend()
    plt.yscale('log')
    fig_stat.add_subplot(2,2,4)
    plt.hist(dB, bins=10, label=f"B_pred - B_real: mean {dB.mean():.5f} std {dB.std():.5f}")
    plt.legend()
    plt.yscale('log')
    plt.savefig(f'{path}/hist_result.png')
    plt.show()
    plt.close()
    model_output = model_output.reshape((N_val, N_val, N_val, 3))    
    #model_output = model_output.permute(2,1,0,3)
    #model_output = model_output.detach().numpy()

    temp_final = temp_final.reshape((N_val, N_val, N_val, 3))
    model_output = model_output.reshape(N_val, N_val, N_val, 3)
    temp_final = temp_final.reshape(N_val, N_val, N_val, 3)

    ax = ['Bx', 'By', 'Bz']
    idxlist=np.linspace(0,N_val-1,5,dtype=int)
    #在不同的z方向上看xy平面的结果
    for i in range(3):
        figure = plt.figure(figsize=(20,30))
        pred = model_output[:,:,:,i].detach().numpy() 
        real = temp_final[:,:,:,i]
        for j in range(5):
            idx=idxlist[j]
            pred_slice = pred[:,:,idx]                       
            real_slice = real[:,:,idx]
            figure.add_subplot(5,3,j*3+1)
            plt.rc('font', size=16)
            X = xx[:,:,idx]
            Y = yy[:,:,idx]
            CS = plt.contourf(X,Y, pred_slice, N_val, cmap='jet')
            plt.colorbar(CS)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'pred field at z={np.round(-L+idx*L*2/N_val,2)}') 
            if(not j==4):
                plt.gca().axes.get_xaxis().set_visible(False)   
            figure.add_subplot(5,3,2+3*j)
            plt.rc('font', size=16)            
            CS = plt.contourf(X, Y, real_slice, N_val, cmap='jet')
            plt.colorbar(CS)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'truth field at z={np.round(-L+idx*L*2/N_val,2)}')
            if(not j==4):
                plt.gca().axes.get_xaxis().set_visible(False)
            plt.gca().axes.get_yaxis().set_visible(False)
            figure.add_subplot(5,3,3*j+3)
            plt.rc('font', size=16)
            CS = plt.contourf(X,Y,(pred_slice-real_slice)/real_slice, N_val, cmap='jet')
            plt.colorbar(CS)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('relative error pred-truth/truth')
            if(not j==4):
                plt.gca().axes.get_xaxis().set_visible(False)
            plt.gca().axes.get_yaxis().set_visible(False)
        plt.savefig(f'{path}/slice_result_z_{ax[i]}.png')
        plt.show()
        plt.close()
    print(f'plot z slices done: {time.time()-st}s')

    #在不同的y方向上看zx平面的结果
    for i in range(3):
        figure = plt.figure(figsize=(20,30))
        pred = model_output[:,:,:,i].detach().numpy()
        real = temp_final[:,:,:,i]
        for j in range(5):
            idx=idxlist[j]
            pred_slice = pred[idx,:,:]                       
            real_slice = real[idx,:,:]
            figure.add_subplot(5,3,j*3+1)
            plt.rc('font', size=16)
            X = xx[idx,:,:]
            Z = zz[idx,:,:]
            CS = plt.contourf(X,Z, pred_slice, N_val, cmap='jet')
            plt.colorbar(CS)
            plt.xlabel('x')
            plt.ylabel('z')
            plt.title(f'pred field at y={np.round(-L+idx*L*2/N_val,2)}')
            if(not j==4):
                plt.gca().axes.get_xaxis().set_visible(False)   
            figure.add_subplot(5,3,j*3+2)
            plt.rc('font', size=16)
            CS = plt.contourf(X,Z, real_slice, N_val, cmap='jet')
            plt.colorbar(CS)
            plt.xlabel('x')
            plt.ylabel('z')
            plt.title(f'truth field at y={np.round(-L+idx*L*2/N_val,2)}')
            if(not j==4):
                plt.gca().axes.get_xaxis().set_visible(False)
            plt.gca().axes.get_yaxis().set_visible(False)   
            figure.add_subplot(5,3,j*3+3)
            plt.rc('font', size=16)
            CS = plt.contourf(X,Z,(pred_slice-real_slice)/real_slice, N_val, cmap='jet')
            plt.colorbar(CS)
            plt.xlabel('x')
            plt.ylabel('z')
            plt.title('relative error pred-truth/truth')
            if(not j==4):
                plt.gca().axes.get_xaxis().set_visible(False)   
            plt.gca().axes.get_yaxis().set_visible(False)
        plt.savefig(f'{path}/slice_result_y_{ax[i]}.png')
        plt.show()
        plt.close()            
    print(f'plot y slices done: {time.time()-st}s')

    #在不同的x方向上看yz平面的结果
    for i in range(3):
        figure = plt.figure(figsize=(20,30))
        pred = model_output[:,:,:,i].detach().numpy()
        real = temp_final[:,:,:,i]
        for j in range(5):
            idx=idxlist[j]
            pred_slice = pred[:,idx,:]
            real_slice = real[:,idx,:]
            figure.add_subplot(5,3,j*3+1)
            plt.rc('font', size=16)
            Y = yy[:,idx,:]
            Z = zz[:,idx,:]
            CS = plt.contourf(Y,Z, pred_slice, N_val, cmap='jet')
            plt.colorbar(CS)
            plt.xlabel('y')
            plt.ylabel('z')
            plt.title(f'pred field at x={np.round(-L+idx*L*2/N_val,decimals=2)}')
            if(not j==4):
                plt.gca().axes.get_xaxis().set_visible(False)   
            figure.add_subplot(5,3,j*3+2)
            plt.rc('font', size=16)
            CS = plt.contourf(Y, Z, real_slice, N_val, cmap='jet')
            plt.colorbar(CS)
            plt.xlabel('y')
            plt.ylabel('z')
            plt.title(f'truth field at x={np.round(-L+idx*L*2/N_val,decimals=2)}')
            if(not j==4):
                plt.gca().axes.get_xaxis().set_visible(False)   
            plt.gca().axes.get_yaxis().set_visible(False)
            figure.add_subplot(5,3,j*3+3)
            plt.rc('font', size=16)
            CS = plt.contourf(Y,Z,(pred_slice-real_slice)/real_slice, N_val, cmap='jet')
            plt.colorbar(CS)
            plt.xlabel('y')
            plt.ylabel('z')
            plt.title('relative error pred-truth/truth')
            if(not j==4):
                plt.gca().axes.get_xaxis().set_visible(False)   
            plt.gca().axes.get_yaxis().set_visible(False)
        plt.savefig(f'{path}/slice_result_x_{ax[i]}.png')
        plt.show()
        plt.close()
    print(f'plot x slices done: {time.time()-st}s')