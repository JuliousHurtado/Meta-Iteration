import sys,os,argparse,time
import numpy as np
import torch

# import utils

tstart=time.time()

# Arguments
parser=argparse.ArgumentParser(description='xxx')
parser.add_argument('--seed',type=int,default=0,help='(default=%(default)d)')
parser.add_argument('--experiment',default='',type=str,required=True,choices=['pmnist'],help='(default=%(default)s)')
parser.add_argument('--approach',default='',type=str,required=True,choices=['sgd','ewc','hat','hat-conv'],help='(default=%(default)s)')
parser.add_argument('--output',default='',type=str,required=False,help='(default=%(default)s)')
parser.add_argument('--nepochs',default=50,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--lr',default=0.003,type=float,required=False,help='(default=%(default)f)')
parser.add_argument('--parameter',type=str,default='',help='(default=%(default)s)')
args=parser.parse_args()
if args.output=='':
    args.output='../res/'+args.experiment+'_'+args.approach+'_'+str(args.seed)+'.txt'
print('='*100)
print('Arguments =')
for arg in vars(args):
    print('\t'+arg+':',getattr(args,arg))
print('='*100)

########################################################################################################################

# Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
use_cuda = torch.cuda.is_available()
if use_cuda: 
    torch.cuda.manual_seed(args.seed)
else: 
    print('[CUDA unavailable]'); sys.exit()

device = torch.device("cuda" if use_cuda else "cpu")

# Args -- Experiment
if args.experiment=='pmnist':
    from pmnist import DatasetGen as dataloader

# Args -- Approach
if args.approach=='sgd':
    from approaches import sgd as approach
elif args.approach=='ewc':
    from approaches import ewc as approach
elif args.approach=='hat' or args.approach=='hat-conv':
    from approaches import hat as approach

# Args -- Network
# if args.experiment=='mnist2' or args.experiment=='pmnist':
#     if args.approach=='hat' or args.approach=='hat-test':
#         from networks import mlp_hat as network
#     else:
#         from networks import mlp as network
# else:
if args.approach=='hat':
    from networks import hat_alexnet as network
elif args.approach=='hat-conv':
    from networks import hat_conv as network
else:
    from networks import alexnet as network

########################################################################################################################

# Load
print('Load data...')
data=dataloader(seed=args.seed, batch_size=64)
taskcla,inputsize = data.taskcla, data.inputsize
print('Input size =',inputsize,'\nTask info =',taskcla)

# Inits
print('Inits...')
net=network.Net(inputsize,taskcla).to(device)
# utils.print_model_report(net)

appr=approach.Appr(net,nepochs=args.nepochs,lr=args.lr,args=args,device=device)
print(appr.criterion)
# utils.print_optimizer_config(appr.optimizer)
print('-'*100)

# Loop tasks
acc=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)
lss=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)
for t,ncla in taskcla:
    print('*'*100)
    print('Task {:2d}'.format(t))
    print('*'*100)

    task_dataloader = data.get(t)

    # Train
    appr.train(t,task_dataloader[t]['train'],task_dataloader[t]['valid'])
    print('-'*100)

    # Test
    for u in range(t+1):
        task_dataloader[u]['test']
        test_loss,test_acc=appr.eval(u,task_dataloader[u]['test'])
        print('>>> Test on task {:2d}: loss={:.3f}, acc={:5.1f}% <<<'.format(u,test_loss,100*test_acc))
        acc[t,u]=test_acc
        lss[t,u]=test_loss

# Done
print('*'*100)
print('Accuracies =')
for i in range(acc.shape[0]):
    print('\t',end='')
    for j in range(acc.shape[1]):
        print('{:5.1f}% '.format(100*acc[i,j]),end='')
    print()
print('*'*100)
print('Done!')

print('[Elapsed time = {:.1f} h]'.format((time.time()-tstart)/(60*60)))

########################################################################################################################
