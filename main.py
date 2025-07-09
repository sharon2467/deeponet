from data import *
from model import *
from train import *
from eval import *
from utils import *
import argparse
import json
#deeponet程序基本上是基于PINN程序修改的，除了model模块完全重写,目前只支持模拟数据
#相较于PINN，deeponet这边能够生成更多种模拟数据，包括三角函数，贝塞尔函数，球谐函数，亥姆霍兹圆线圈，亥姆霍兹方线圈，两圆一方
#deeponet结构和论文内的相同，linear采用线性层，conv采用卷积层，推荐使用linear
#标准化函数
def data_standardization(data,config,dataname):
    if(config[dataname+'_standard']==1):
        config['mean_'+dataname] = data.mean(axis=0).tolist()
        config['std_'+dataname]  = data.std(axis=0).tolist()
        data = ((data - config['mean_'+dataname])/config['std_'+dataname])
    else:
        config['mean_'+dataname] = 0
        config['std_'+dataname]  = 1
    np.save(f"{config['path']}/{dataname}.npy", data)
    return data,config
#由于没有使用subparser，所以不是所有输入的参数都会被用到
parser = argparse.ArgumentParser(description='DeepONet field ediction')
parser.add_argument('--mode', type=str, metavar='--', choices=['train', 'eval'],default='train',help='train or eval the model')
parser.add_argument('--eval_path',type=str,metavar='--',help='path to the model you want to evaluate')
parser.add_argument('--logdir', type=str, default='./log/', metavar='--',
                    help='log dir')
parser.add_argument('--model_type',default='DeepONet',type=str, metavar='--', choices=['DeepONet','DeepONet_Conv'],help='type of the model you want to use,deeponet_conv is unstable now')
parser.add_argument('--order',type=int,metavar='--',default=1,help='order of the field,the solution to Laplace equation have various orders')
parser.add_argument('--sample', type=int, metavar='--', default=200, help='number of samples')
parser.add_argument('--experiment', type=str, default='training', metavar='--',
                    help='name of the experiment you want to do, like scan different learning rate, scan different sample points')
parser.add_argument('--device', type=str, default='cpu', metavar='--', choices=['cpu', 'cuda:0'],
                    help='device type, cpu or cuda:0')
parser.add_argument('--adjust_lr', type=int, default=0, metavar='--', choices=[0, 1],
                    help='whether adjust the lr during training, 0 means no, 1 means yes')
parser.add_argument('--N', type=int, default=6, metavar='--',
                    help='number of sample points per edge')
parser.add_argument('--rotation_on',type=int,default=0,metavar='--',choices=[0,1],help='rotation on the field or not')
parser.add_argument('--order_on',type=float,default=0,metavar='--',help='order on the field or not')
parser.add_argument('--phase_on',type=float,default=0,metavar='--',help='phase on the field or not')
#middlelayer是把trunk和branch混合的layer
parser.add_argument('--num_mid', type=int, default=32, metavar='--',help='number of neurals in the middle layer')
parser.add_argument('--radius', type=float, default=1, metavar='--',
                    help='radius of the coils')
parser.add_argument('--length', type=float, default=1, metavar='--',
                    help='side length of the area that you want to predict')
parser.add_argument('--units', type=int, default=32, metavar='--', 
                    help='number of neurals in a network layer')
parser.add_argument('--Nep', type=int, default=10000, metavar='--', 
                    help='number of epochs')
#deeponet相比于pinn的一个重大差别就是需要固定位置传感器上的磁场数据作为输入，而不是损失计算时才使用，Bdata是传感器上的磁场数据,posdata是训练采样的位置（注意不是传感器位置！），labels是训练采样的磁场数据
#相比pinn只需要标准化posdata和labels，deeponet还需要标准化Bdata，因为Bdata是作为输入而不是损失函数而出现的，也就是说pinn仅有posdata一个输入，deeponet有Bdata和posdata两个
#这使得其可以在训练时使用不同的Bdata和posdata，从而在磁场发生改变时可以继续预测而无需重新训练模型
#作为代价，其需要大量的数据才能保证可靠度。且初版deeponet无物理损失加强，完全靠数据驱动
parser.add_argument('--Bdata_standard',type=int,default=0,metavar='--',choices=[0,1],help='perform standardization on data or not')
parser.add_argument('--posdata_standard',type=int,default=0,metavar='--',choices=[0,1],help='perform standardization on data or not')
parser.add_argument('--labeldata_standard',type=int,default=0,metavar='--',choices=[0,1],help='perform standardization on data or not')
parser.add_argument('--Btype', type=str, default='circle', metavar='--', choices=['cylinder','cubic','spherical','rectangle','circle','reccirc'],
                    help='which type field you want to generate')
parser.add_argument('--dx', type=float, default=9999, metavar='--',help='the distance in x direction of the two helmholtz coils')
parser.add_argument('--dy', type=float, default=9999, metavar='--',help='the distance in y direction of the two helmholtz coils')
parser.add_argument('--dz', type=float, default=9999, metavar='--',help='the distance in z direction of the two helmholtz coils')
parser.add_argument('--radius1', type=float, default=9999, metavar='--',help='the radius of the first helmholtz coil')
parser.add_argument('--radius2', type=float, default=9999, metavar='--',help='the radius of the second helmholtz coil')
parser.add_argument('--a', type=float, default=9999, metavar='--',help='x length of the rectangle')
parser.add_argument('--b', type=float, default=9999, metavar='--',help='y length of the rectangle')
parser.add_argument('--Iz', type=float, default=1, metavar='--',help='z Intensity only used in reccirc')
parser.add_argument('--Ix', type=float, default=1, metavar='--',help='x Intensity only used in reccirc')
parser.add_argument('--Iy', type=float, default=1, metavar='--',help='y Intensity only used in reccirc')
parser.add_argument('--sets', type=int, default=2, metavar='--',help='number of sets of samples')
#随机化是deeponet中一个很有效的措施，通过把来源于不同磁场的样本混合，可以避免loss振荡，获得更低的loss
parser.add_argument('--randomization', type=int, default=0, metavar='--', choices=[0, 1],
                    help='randomization of the data, 0 means no, 1 means yes')
args = parser.parse_args()
if((args.dx==9999) and (args.dy==9999) and (args.dz==9999) and (args.a)==9999 and (args.b)==9999):
    args.dx = args.radius*2
    args.dy = args.radius*2
    args.dz = args.radius*2
    args.a=args.radius*2
    args.b=args.radius*2
if((args.radius1==9999) and (args.radius2==9999)):
    args.radius1 = args.radius
    args.radius2 = args.radius
if(args.mode=='train'):
    config = {}    
    config.update(vars(args))
    config['logdir']    = args.logdir + '/' + args.experiment
    path = mkdir(config['logdir'])
    config['path'] = path
    field = Data_Generator(config)
    if(config['model_type']=='DeepONet'):
        probe_pos=field.pos_generation()
        model=DeepONet_Linear(config['units'],config['num_mid'],probe_pos)
        train_Bdata,train_posdata,train_labels=field.data_generation(probe_pos,config['Btype'])
        test_Bdata,test_posdata,test_labels=field.data_generation(probe_pos,config['Btype'])
        
    elif(config['model_type']=='DeepONet_Conv'):
        #如果是卷积网络，probe_pos需要在meshgrid_generation中生成。这意味着所有数据点都必须在均匀的网格上。
        train_Bdata,train_posdata,train_labels,probe_pos=field.meshgrid_generation(config['Btype'])
        probe_pos=torch.tensor(probe_pos,dtype=torch.float32)
        model=DeepONet_Conv(config['units'],config['num_mid'],config['N'],probe_pos)
        test_Bdata,test_posdata,test_labels,_=field.meshgrid_generation(config['Btype'])
        np.save(f"{path}/probe_pos.npy", probe_pos)
    #数据预处理
    train_Bdata,config=data_standardization(train_Bdata,config,'Bdata')
    train_pos_data,config=data_standardization(train_posdata,config,'posdata')
    train_labels,config=data_standardization(train_labels,config,'labeldata') 
    test_Bdata=(test_Bdata-config['mean_Bdata'])/config['std_Bdata']
    test_posdata=(test_posdata-config['mean_posdata'])/config['std_posdata']
    test_labels=(test_labels-config['mean_labeldata'])/config['std_labeldata']  
    train_Bdata = torch.tensor(train_Bdata, dtype=torch.float32)
    train_posdata = torch.tensor(train_posdata, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.float32)
    test_Bdata = torch.tensor(test_Bdata, dtype=torch.float32)
    test_posdata = torch.tensor(test_posdata, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.float32)

    with open(f"{path}/config.json", 'w') as config_file:
        config_file.write( json.dumps(config, indent=4) )
    #exit()
    train(model,train_Bdata,train_posdata,train_labels,test_Bdata,test_posdata,test_labels,config)
    
if(args.mode=='eval'):
    with open(f"{args.eval_path}/config.json", 'r') as config_file:
        config = json.load(config_file)
    field = Data_Generator(config['length'],config['N'],config['order'],config['samples'],config['cons_l'],config['rotation_on'],config['order_on'],config['phase_on'])
    if(config['model_type']=='DeepONet'):
        model=DeepONet_Linear(config['units'],config['num_mid'],np.load(f"{args.eval_path}/probe_pos.npy"))
    else:
        model=DeepONet_Conv(config['units'],config['num_mid'],config['N'])
    train_Bdata=np.load(f"{args.eval_path}/train_Bdata.npy")
    train_pos_data=np.load(f"{args.eval_path}/train_posdata.npy")
    train_labels=np.load(f"{args.eval_path}/train_labels.npy")
    model.load_state_dict(torch.load(f'{args.eval_path}/best_model.pt'))
    model.eval()
Eval(model,config,field)

