import torch
import numpy as np
from model import *
from data import *
import time
import matplotlib.pyplot as plt
from torch import optim

def train(model,train_Bdata,train_posdata, train_labels, test_Bdata,test_posdata, test_labels, config):
    Nep    = config['Nep']
    device = config['device']
    path   = config['path'] 
    adjust = config['adjust_lr']
    randomization=config['randomization']
    lr=0.005
    model  = model.to(device)
    optimizer1 = optim.AdamW(model.parameters(), lr)
    optimizer2=optim.LBFGS(model.parameters(),lr)
    scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer1,patience=800)
    criterion = PINN_Loss(256,config['length'],device,0,1,0)
    loss_f_l = []
    loss_u_l = []
    loss_cross_l = []
    loss_BC_div_l = []
    loss_BC_cul_l = []
    loss_l = []
    test_loss_l = []
    epoch = []
    mini_loss = 100000000
    best_model = model
    best_ep = 0
    train_Bdata = train_Bdata.requires_grad_(True).to(device)
    train_posdata = train_posdata.requires_grad_(True).to(device)
    train_labels = train_labels.requires_grad_(True).to(device)
    test_Bdata=test_Bdata.to(device)
    test_posdata=test_posdata.to(device)
    test_labels=test_labels.to(device)     
    st = time.time()
    exitflag=0
    patience=3000
    def closure():
        optimizer2.zero_grad()
        train_Bdata_slice,train_posdata_slice,train_labels_slice=train_data_generation(train_Bdata,train_posdata,train_labels,config,model,randomization)
        loss_f, loss_u, loss_cross, loss_BC_div, loss_BC_cul, loss = criterion.loss(train_posdata_slice,train_Bdata_slice,train_labels_slice,model)
        loss.backward()
        return loss
    for ep in range(Nep):
        model.train()
        if(ep<Nep*0.9 and exitflag<patience+20):
            optimizer1.zero_grad()
            train_Bdata_slice,train_posdata_slice,train_labels_slice=train_data_generation(train_Bdata,train_posdata,train_labels,config,model,randomization)
            loss_f, loss_u, loss_cross, loss_BC_div, loss_BC_cul, loss = criterion.loss(train_posdata_slice,train_Bdata_slice,train_labels_slice,model)
            loss.backward()
            optimizer1.step()
            if(adjust):
                scheduler.step(loss)
        else:
            optimizer2.step(closure=closure)
            train_Bdata_slice,train_posdata_slice,train_labels_slice=train_data_generation(train_Bdata,train_posdata,train_labels,config,model,randomization)
            loss_f, loss_u, loss_cross, loss_BC_div, loss_BC_cul, loss = criterion.loss(train_posdata_slice,train_Bdata_slice,train_labels_slice,model)
        if(ep%100==0):
            epoch.append(ep)
            loss_f_l.append(loss_f.item())
            loss_u_l.append(loss_u.item())
            loss_cross_l.append(loss_cross.item())
            loss_BC_div_l.append(loss_BC_div.item())
            loss_BC_cul_l.append(loss_BC_cul.item())
            loss_l.append(loss.item())

            with torch.no_grad():
                model.eval()
                idx1=np.random.randint(0,test_Bdata.size()[0])
                if(config['model_type']=='DeepONet'):
                    test_pred = model(test_Bdata[idx1,:],test_posdata[:,:,idx1])
                elif(config['model_type']=='DeepONet_Conv'):
                    test_pred = model(test_Bdata[:,:,:,idx1],test_posdata[:,:,idx1])
                print(f'max:{torch.max(test_pred-test_labels[:,:,idx1])}')
                test_loss = torch.mean(torch.square(test_pred-test_labels[:,:,idx1]))
                test_loss_l.append(test_loss.item())

                #if(adjust):
                    #lr_adjust(test_loss, optimizer1)            
                if(mini_loss>test_loss and exitflag<patience and ep>Nep*0.9):
                    torch.save(model.state_dict(), f'{path}/best_model.pt')
                    mini_loss = test_loss
                    best_model = model
                    best_ep = ep
                    exitflag=0
                else:
                    exitflag=exitflag+1
            model.train()
            if(test_loss<0.0000001 or exitflag>patience+5):
                print('early stop!!!')
                break

        if(ep%100==0):
            print(f'===>>> ep: {ep}')
            print(f'time used: {time.time()-st:.2f}s, time left: {(time.time()-st)/(ep+1)*Nep-(time.time()-st):.2f}s')
            print(f'total loss: {loss:.7f}, test loss: {test_loss:.7f}')
            print(f'loss div: {loss_f:.7f}, loss B: {loss_u:.7f}, loss cul: {loss_cross:.7f}')
            print(f'loss BC div: {loss_BC_div:.7f}, loss BC cul: {loss_BC_cul:.7f}')
                
    print(f'best loss at ep: {best_ep}, best_loss: {mini_loss:.7f}')
    print(f'total time used: {time.time()-st:.2f}s')
    plt.plot(epoch, loss_f_l, label='loss div')
    plt.plot(epoch, loss_u_l, label='loss B')
    plt.plot(epoch, loss_cross_l, label='loss cul')
    plt.plot(epoch, loss_BC_div_l, label='loss BC div')
    plt.plot(epoch, loss_BC_cul_l, label='loss BC cul')    
    plt.plot(epoch, loss_l, label='total loss')
    plt.plot(epoch, test_loss_l, label='test loss')
    plt.scatter(best_ep, mini_loss.to('cpu').item(), label='test best loss', marker='*')
    plt.legend()
    plt.yscale('log')
    plt.savefig(f'{path}/loss'+'.png')  
    plt.show()
    plt.close()
    np.save(f'{path}/loss'+'.npy',        np.array(loss_l))
    np.save(f'{path}/loss_test'+'.npy',   np.array(test_loss_l))

    return best_model
def train_data_generation(train_Bdata,train_posdata,train_labels,config,model,randomization):
    if(randomization==0):
        idx=np.random.randint(train_Bdata.size()[0])        
        train_posdata_slice=train_posdata[:,:,idx]
        train_labels_slice=train_labels[:,:,idx]
        #print(train_posdata_slice.size())
        if(config['model_type']=='DeepONet'):
            train_Bdata_slice=train_Bdata[idx,:]
            train_posdata_slice=torch.cat((train_posdata_slice,torch.tensor(model.probes_pos)),axis=0)
            #print(train_posdata_slice.size())
            train_labels_slice=torch.cat((train_labels_slice,torch.reshape(train_Bdata_slice,(-1,3))),axis=0)
        elif(config['model_type']=='DeepONet_Conv'):
            train_Bdata_slice=train_Bdata[:,:,:,idx]
            train_posdata_slice=torch.cat((train_posdata_slice,model.probes_pos),axis=0)
            for i in range(6):
                a=torch.zeros(model.n**2,3)
                for j in range(3):
                    a[:,j]=train_Bdata_slice[:,:,i*3+j].reshape(-1)
                train_labels_slice=torch.cat((train_labels_slice,a),axis=0)
    else:
        idx1=np.random.randint(0,train_Bdata.size()[0],config['sample'])
        idx2=np.random.randint(0,train_posdata.size()[0],config['sample'])
        train_Bdata_slice=train_Bdata[idx1,:]
        train_posdata_slice=train_posdata[idx2,:,idx1]
        train_labels_slice=train_labels[idx2,:,idx1]
    #print(train_posdata_slice.size())
    return train_Bdata_slice,train_posdata_slice,train_labels_slice