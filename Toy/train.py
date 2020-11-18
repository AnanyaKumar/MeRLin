import  numpy as np
import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch import optim
import  copy
import  os
import  random

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
# set the hyper-parameters

sigma = 0.01
num_source = 10000
num_target = 20
d = 500
lr = 0.03
weight_decay = 1e-3
num_step = 200000
display_norm = True
display_test_loss = True
display_interval = 1000


# 'joint-training', 'meta-learning', 'algorithm1', 'source', 'target','joint-training+fine-tuning',
#algorithm = ['joint-training+fine-tuning']
algorithm = ['algorithm1']
#,'joint-training', 'meta-learning', 'algorithm1', 'joint-training+fine-tuning', 'fine_tune']
algorithm = ['joint-training','meta-learning-shuffle', 'algorithm1']

print('algorithm:   ',algorithm)
print('delta:       ',delta)
print('epsilon:     ',epsilon)
print('num_source:  ',num_source)
print('num_target:  ',num_target)
print('lr:          ',lr)
print('weight_decay:',weight_decay)
print('num_step:    ',num_step)


random.seed(101)
np.random.seed(101)
torch.manual_seed(101)   


def source_data_generate(num,d):
    
    data = np.random.randint(-1, high=2, size=(num,d), dtype='l').astype('float32')
    arr = [1,-1,0]
    ar = np.random.choice(arr, num, p=[0.25,0.25, 0.5])
    br = copy.deepcopy(ar)
    br[np.random.rand(num) > 0.5] *= -1
    data[:,0:2] = np.concatenate([ar.reshape(num,1),br.reshape(num,1)], axis = -1)
    label = (inx[:,0] == 0).astype('int64')
    data = torch.from_numpy(data.astype('float32')).cuda()
    label = torch.from_numpy(label.astype('int64')).cuda()

    return  data, label


def target_data_generate(num,d):

    data = np.random.randint(-1, high=2, size=(num,d), dtype='l').astype('float32')
    arr = [1,-1,0]
    ar = np.random.choice(arr, num, p=[0.25,0.25, 0.5])
    data[:,1] = ar.reshape(num,1)
    data[:,0] = 0
    label = (inx[:,1] == 0).astype('int64')
    data = torch.from_numpy(data.astype('float32')).cuda()
    label = torch.from_numpy(label.astype('int64')).cuda()   

def weights_init(sigma,m):                                                                
        nn.init.normal_(m.weight.data, 0.0, sigma)                                                  
        #nn.init.constant_(m.weight.data, 0)   

def target(sigma,num_source,num_target,d,lr,weight_decay,num_step,display_norm,display_test_loss,display_interval):
    
     
    target_train, target_label_train = target_data_generate(num_target,d)
    
    w = nn.Linear(500,2,bias=False).cuda()
    a_S = nn.Linear(2,2,bias=False).cuda()
    a_T = nn.Linear(2,2,bias=False).cuda()

    optimizer = torch.optim.SGD(w.parameters(),lr=lr,weight_decay=weight_decay,momentum=0)
    optimizer1 = torch.optim.SGD(a_S.parameters(),lr=lr,weight_decay=weight_decay,momentum=0)
    optimizer2 = torch.optim.SGD(a_T.parameters(),lr=lr,weight_decay=weight_decay,momentum=0)

    creterion = torch.nn.CrossEntropyLoss()

    a_T_ts = list(a_T.named_parameters())[0][-1]
    a_S_ts = list(a_S.named_parameters())[0][-1]
    w_ts = list(w.named_parameters())[0][-1]

    target_test, target_label_test = target_data_generate(20000,d)
    

    for i in range(num_step):
        loss_target = creterion(a_T(F.relu(w(target_train))), target_label_train) 

        optimizer.zero_grad()
        optimizer2.zero_grad()
        loss = loss_target
        loss.backward()
        optimizer.step()
        optimizer2.step()
        
        if i % display_interval == 0:
            if display_norm and not display_test_loss:
                print('iter: %6d | a_T: %.3f | a_S: %.3f | w: %.3f'%(
                                                 i,
                                                 torch.norm(a_T_ts).data.cpu().numpy(), 
                                                 torch.norm(a_S_ts).data.cpu().numpy(),
                                                 torch.norm(w_ts).data.cpu().numpy()))

            if not display_norm and display_test_loss:
                print('iter: %6d | test_acc: %.6f'%(
                                                 i,
                                                 torch.sum(torch.max(a_T(F.relu(w(target_test))), dim = -1)[-1] == target_label_test).data.cpu().numpy())
            
            if display_norm and display_test_loss:
                print('iter: %6d | a_T: %.3f | a_S: %.3f | w: %.3f | test_acc: %.6f'%(
                                                 i,
                                                 torch.norm(a_T_ts).data.cpu().numpy(), 
                                                 torch.norm(a_S_ts).data.cpu().numpy(),
                                                 torch.norm(w_ts).data.cpu().numpy(),
                                                 torch.sum(torch.max(a_T(F.relu(w(target_test))), dim = -1)[-1] == target_label_test).data.cpu().numpy())

    print('training loss:     ',loss_target.data.cpu().numpy())

    loss_target_test = torch.sum(torch.max(a_T(F.relu(w(target_test))), dim = -1)[-1] == target_label_test)
    print('test loss:         ',loss_target_test.data.cpu().numpy())

    
    
def joint_training(sigma,num_source,num_target,d,lr,weight_decay,num_step,display_norm,display_test_loss,display_interval):
    
    # generate data
     
    source_train, source_label = source_data_generate(b_S,sigma,epsilon,num_source,d)
    target_train, target_label_train = target_data_generate(num_target,d)
    
    # set the model
    # w 2*500
    # a_S and a_T 1*2 
    w = nn.Linear(500,2,bias=False).cuda()
    a_S = nn.Linear(2,2,bias=False).cuda()
    a_T = nn.Linear(2,2,bias=False).cuda()

    # init the model
    # you can choose gaussian or constant
    '''
    def weights_init(m):                                                                
        nn.init.normal_(m.weight.data, 0.0, 0.01)                                                  
        #nn.init.constant_(m.weight.data, 0)                          

    w.apply(weights_init)
    a_S.apply(weights_init) 
    a_T.apply(weights_init) 
    '''

    # set the optimizer 
    optimizer = torch.optim.SGD(w.parameters(),lr=lr,weight_decay=weight_decay,momentum=0)
    optimizer1 = torch.optim.SGD(a_S.parameters(),lr=lr,weight_decay=weight_decay,momentum=0)
    optimizer2 = torch.optim.SGD(a_T.parameters(),lr=lr,weight_decay=weight_decay,momentum=0)

    # using mse loss
    creterion = torch.nn.CrossEntropyLoss()

    # set the data for training
    source_train = source_train.T
    target_train = target_train.T


    a_T_ts = list(a_T.named_parameters())[0][-1]
    a_S_ts = list(a_S.named_parameters())[0][-1]
    w_ts = list(w.named_parameters())[0][-1]

    # generate test data
    source_test, source_label_test = source_data_generate(b_S,sigma,epsilon,20000,d)
    target_test, target_label_test = target_data_generate(20000,d)

    source_test = source_test.T
    

    # train the model
    for i in range(num_step):
        loss_source = creterion(a_S(w(source_train)), source_label) 
        loss_target = creterion(a_T(F.relu(w(target_train))), target_label_train) 

        optimizer.zero_grad()
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        loss = loss_source + loss_target
        loss.backward()
        optimizer.step()
        optimizer1.step()
        optimizer2.step()
        
        if i % display_interval == 0:
            if display_norm and not display_test_loss:
                print('iter: %6d | a_T: %.3f | a_S: %.3f | w: %.3f'%(
                                                 i,
                                                 torch.norm(a_T_ts).data.cpu().numpy(), 
                                                 torch.norm(a_S_ts).data.cpu().numpy(),
                                                 torch.norm(w_ts).data.cpu().numpy()))

            if not display_norm and display_test_loss:
                print('iter: %6d | test_loss: %.6f | norm_of_difference: %.6f'%(
                                                 i,
                                                 torch.sum(torch.max(a_T(F.relu(w(target_test))), dim = -1)[-1] == target_label_test).data.cpu().numpy(),
                                                 torch.norm(list(a_T.named_parameters())[0][-1].mm(list(w.named_parameters())[0][-1]) - b_T).data.cpu().numpy()))
            
            if display_norm and display_test_loss:
                print('iter: %6d | a_T: %.3f | a_S: %.3f | w: %.3f | test_loss: %.6f | norm_of_difference: %.6f'%(
                                                 i,
                                                 torch.norm(a_T_ts).data.cpu().numpy(), 
                                                 torch.norm(a_S_ts).data.cpu().numpy(),
                                                 torch.norm(w_ts).data.cpu().numpy(),
                                                 torch.sum(torch.max(a_T(F.relu(w(target_test))), dim = -1)[-1] == target_label_test).data.cpu().numpy(),
                                                 torch.norm(list(a_T.named_parameters())[0][-1].mm(list(w.named_parameters())[0][-1]) - b_T).data.cpu().numpy()))

    # print training loss     
    print('training loss:     ',loss_source.data.cpu().numpy(),loss_target.data.cpu().numpy())

    # calculate test loss 
    loss_source_test = creterion(a_S(w(source_test)), source_label_test) 
    loss_target_test = creterion(a_T(F.relu(w(target_test))), target_label_test)
    print('test loss:         ',loss_source_test.data.cpu().numpy(), loss_target_test.data.cpu().numpy())


    # calculate \|aw - b\|
    norm_tgt = torch.norm(list(a_T.named_parameters())[0][-1].mm(list(w.named_parameters())[0][-1]) - b_T)
    norm_src = torch.norm(list(a_S.named_parameters())[0][-1].mm(list(w.named_parameters())[0][-1]) - b_S)
    print('norm of difference:',norm_src.data.cpu().numpy(),norm_tgt.data.cpu().numpy())
    
def meta_learning(sigma,num_source,num_target,d,lr,weight_decay,num_step,display_norm,display_test_loss,display_interval):
    
     
    source_train, source_label = source_data_generate(b_S,sigma,epsilon,num_source,d)
    target_train, target_label_train = target_data_generate(num_target,d)
    target_val, target_label_val = target_data_generate(b_T,sigma,num_target,d)

    w = nn.Linear(500,2,bias=False).cuda()
    a_S = nn.Linear(2,2,bias=False).cuda()
    a_T = nn.Linear(2,2,bias=False).cuda() 

    def weights_init(m):                                                                
        nn.init.normal_(m.weight.data, 0.0, 0.01)                                                  
        #nn.init.constant_(m.weight.data, 0)                          

    w.apply(weights_init)
    a_S.apply(weights_init) 
    a_T.apply(weights_init) 

    optimizer = torch.optim.SGD(w.parameters(),lr=lr,weight_decay=weight_decay,momentum=0)
    optimizer1 = torch.optim.SGD(a_S.parameters(),lr=lr,weight_decay=weight_decay,momentum=0)

    creterion = torch.nn.CrossEntropyLoss()

    source_train = source_train.T 
    
    target_val = target_val.T 

    a_T_ts = list(a_T.named_parameters())[0][-1]
    a_S_ts = list(a_S.named_parameters())[0][-1]
    w_ts = list(w.named_parameters())[0][-1]

    source_test, source_label_test = source_data_generate(b_S,sigma,epsilon,20000,d)
    target_test, target_label_test = target_data_generate(20000,d)

    source_test = source_test.T 
     


    for i in range(num_step):
        f_T = w(target_train)
        f_T_val = w(target_val)

        loss_source = creterion(a_S(w(source_train)), source_label) 
        loss_meta = creterion(target_label_train.T.mm(torch.inverse(f_T.T.mm(f_T) + weight_decay * torch.eye(2).cuda() ).mm(f_T.T).T).mm(f_T_val.T).T, target_label_val) 

        optimizer.zero_grad()
        optimizer1.zero_grad()
        loss = loss_source + 1.0 * loss_meta 
        loss.backward()
        optimizer.step()
        optimizer1.step()
        
        f_T_test = w(target_test)

        if i % display_interval == 0:
            if display_norm and not display_test_loss:
                print('iter: %6d | a_T: %.3f | a_S: %.3f | w: %.3f'%(
                                                 i,
                                                 torch.norm(a_T_ts).data.cpu().numpy(), 
                                                 torch.norm(a_S_ts).data.cpu().numpy(),
                                                 torch.norm(w_ts).data.cpu().numpy()))

            if not display_norm and display_test_loss:
                print('iter: %6d | test_loss: %.6f | norm_of_difference: %.6f'%(
                                                 i,
                                                 creterion(target_label_train.T.mm(torch.inverse(f_T.T.mm(f_T) + weight_decay
                * torch.eye(2).cuda()).mm(f_T.T).T).mm(f_T_test.T).T, target_label_test).data.cpu().numpy(),
                                                 torch.norm(target_label_train.T.mm(torch.inverse(f_T.T.mm(f_T) + weight_decay
                * torch.eye(2).cuda()).mm(f_T.T).T).mm(list(w.named_parameters())[0][-1]) - b_T).data.cpu().numpy()))
            
            if display_norm and display_test_loss:
                print('iter: %6d | a_T: %.3f | a_S: %.3f | w: %.3f | test_loss: %.6f | norm_of_difference: %.6f'%(
                                                 i,
                                                 torch.norm(a_T_ts).data.cpu().numpy(), 
                                                 torch.norm(a_S_ts).data.cpu().numpy(),
                                                 torch.norm(w_ts).data.cpu().numpy(),
                                                 creterion(target_label_train.T.mm(torch.inverse(f_T.T.mm(f_T) + weight_decay
                * torch.eye(2).cuda()).mm(f_T.T).T).mm(f_T_test.T).T, target_label_test).data.cpu().numpy(),
                                                 torch.norm(target_label_train.T.mm(torch.inverse(f_T.T.mm(f_T) + weight_decay
                * torch.eye(2).cuda()).mm(f_T.T).T).mm(list(w.named_parameters())[0][-1]) - b_T).data.cpu().numpy()))


    print('training loss:     ',loss_source.data.cpu().numpy(),loss_meta.data.cpu().numpy())

    f_T_test = w(target_test)

    loss_source_test = creterion(a_S(w(source_test)), source_label_test) 
    loss_target_test = creterion(target_label_train.T.mm(torch.inverse(f_T.T.mm(f_T) + weight_decay
                * torch.eye(2).cuda()).mm(f_T.T).T).mm(f_T_test.T).T, target_label_test)

    print('test loss:         ',loss_source_test.data.cpu().numpy(), loss_target_test.data.cpu().numpy())
    dev_tgt = torch.norm(target_label_train.T.mm(torch.inverse(f_T.T.mm(f_T) + weight_decay
                * torch.eye(2).cuda()).mm(f_T.T).T).mm(list(w.named_parameters())[0][-1]) - b_T)

    dev_src = torch.norm(list(a_S.named_parameters())[0][-1].mm(list(w.named_parameters())[0][-1]) - b_S)
    print('norm of difference:',dev_src.data.cpu().numpy(),dev_tgt.data.cpu().numpy())
    
def meta_learning_shuffle(sigma,num_source,num_target,d,lr,weight_decay,num_step,display_norm,display_test_loss,display_interval):
    
     
    source_train, source_label = source_data_generate(b_S,sigma,epsilon,num_source,d)
    target_train, target_label_train = target_data_generate(num_target,d)
    target_val, target_label_val = target_data_generate(b_T,sigma,num_target,d)

    w = nn.Linear(500,2,bias=False).cuda()
    a_S = nn.Linear(2,2,bias=False).cuda()
    a_T = nn.Linear(2,2,bias=False).cuda() 

    def weights_init(m):                                                                
        nn.init.normal_(m.weight.data, 0.0, 0.01)                                                  
        #nn.init.constant_(m.weight.data, 0)                          

    w.apply(weights_init)
    a_S.apply(weights_init) 
    a_T.apply(weights_init) 

    optimizer = torch.optim.SGD(w.parameters(),lr=lr,weight_decay=weight_decay,momentum=0)
    optimizer1 = torch.optim.SGD(a_S.parameters(),lr=lr,weight_decay=weight_decay,momentum=0)

    creterion = torch.nn.CrossEntropyLoss()

    source_train = source_train.T 
    
    target_val = target_val.T 

    a_T_ts = list(a_T.named_parameters())[0][-1]
    a_S_ts = list(a_S.named_parameters())[0][-1]
    w_ts = list(w.named_parameters())[0][-1]

    source_test, source_label_test = source_data_generate(b_S,sigma,epsilon,20000,d)
    target_test, target_label_test = target_data_generate(20000,d)

    source_test = source_test.T 
     
    target_all = torch.cat([target_train,target_val], dim = 0)

    for i in range(num_step):
        target_train = target_all[0:num_target]
        target_val = target_all[num_target:]
        f_T = w(target_train)
        f_T_val = w(target_val)

        loss_source = creterion(a_S(w(source_train)), source_label) 
        loss_meta = creterion(target_label_train.T.mm(torch.inverse(f_T.T.mm(f_T) + weight_decay * torch.eye(2).cuda() ).mm(f_T.T).T).mm(f_T_val.T).T, target_label_val) 

        optimizer.zero_grad()
        optimizer1.zero_grad()
        loss = loss_source + 1.0 * loss_meta 
        loss.backward()
        optimizer.step()
        optimizer1.step()
        
        f_T_test = w(target_test)

        if i % display_interval == 0:
            if display_norm and not display_test_loss:
                print('iter: %6d | a_T: %.3f | a_S: %.3f | w: %.3f'%(
                                                 i,
                                                 torch.norm(a_T_ts).data.cpu().numpy(), 
                                                 torch.norm(a_S_ts).data.cpu().numpy(),
                                                 torch.norm(w_ts).data.cpu().numpy()))

            if not display_norm and display_test_loss:
                print('iter: %6d | test_loss: %.6f | norm_of_difference: %.6f'%(
                                                 i,
                                                 creterion(target_label_train.T.mm(torch.inverse(f_T.T.mm(f_T) + weight_decay
                * torch.eye(2).cuda()).mm(f_T.T).T).mm(f_T_test.T).T, target_label_test).data.cpu().numpy(),
                                                 torch.norm(target_label_train.T.mm(torch.inverse(f_T.T.mm(f_T) + weight_decay
                * torch.eye(2).cuda()).mm(f_T.T).T).mm(list(w.named_parameters())[0][-1]) - b_T).data.cpu().numpy()))
            
            if display_norm and display_test_loss:
                print('iter: %6d | a_T: %.3f | a_S: %.3f | w: %.3f | test_loss: %.6f | norm_of_difference: %.6f'%(
                                                 i,
                                                 torch.norm(a_T_ts).data.cpu().numpy(), 
                                                 torch.norm(a_S_ts).data.cpu().numpy(),
                                                 torch.norm(w_ts).data.cpu().numpy(),
                                                 creterion(target_label_train.T.mm(torch.inverse(f_T.T.mm(f_T) + weight_decay
                * torch.eye(2).cuda()).mm(f_T.T).T).mm(f_T_test.T).T, target_label_test).data.cpu().numpy(),
                                                 torch.norm(target_label_train.T.mm(torch.inverse(f_T.T.mm(f_T) + weight_decay
                * torch.eye(2).cuda()).mm(f_T.T).T).mm(list(w.named_parameters())[0][-1]) - b_T).data.cpu().numpy()))


    print('training loss:     ',loss_source.data.cpu().numpy(),loss_meta.data.cpu().numpy())

    f_T_test = w(target_test)

    loss_source_test = creterion(a_S(w(source_test)), source_label_test) 
    loss_target_test = creterion(target_label_train.T.mm(torch.inverse(f_T.T.mm(f_T) + weight_decay
                * torch.eye(2).cuda()).mm(f_T.T).T).mm(f_T_test.T).T, target_label_test)

    print('test loss:         ',loss_source_test.data.cpu().numpy(), loss_target_test.data.cpu().numpy())
    dev_tgt = torch.norm(target_label_train.T.mm(torch.inverse(f_T.T.mm(f_T) + weight_decay
                * torch.eye(2).cuda()).mm(f_T.T).T).mm(list(w.named_parameters())[0][-1]) - b_T)

    dev_src = torch.norm(list(a_S.named_parameters())[0][-1].mm(list(w.named_parameters())[0][-1]) - b_S)
    print('norm of difference:',dev_src.data.cpu().numpy(),dev_tgt.data.cpu().numpy())

def algorithm1(sigma,num_source,num_target,d,lr,weight_decay,num_step,display_norm,display_test_loss,display_interval):
    
     
    source_train, source_label = source_data_generate(b_S,sigma,epsilon,num_source,d)
    target_train, target_label_train = target_data_generate(num_target,d)
    target_val, target_label_val = target_data_generate(b_T,sigma,num_target,d)
    
    w = nn.Linear(500,2,bias=False).cuda() 
    a_S = nn.Linear(2,2,bias=False).cuda() 
    a_T = nn.Linear(2,2,bias=False).cuda() 

    def weights_init(m):                                                                
        nn.init.normal_(m.weight.data, 0.0, 0.01)                                                  
        #nn.init.constant_(m.weight.data, 0)                          

    w.apply(weights_init)
    a_S.apply(weights_init) 
    a_T.apply(weights_init) 

    optimizer = torch.optim.SGD(w.parameters(),lr=lr,weight_decay=weight_decay,momentum=0)
    optimizer1 = torch.optim.SGD(a_S.parameters(),lr=lr,weight_decay=weight_decay,momentum=0)

    creterion = torch.nn.CrossEntropyLoss()

    source_train = source_train.T 
    

    a_T_ts = list(a_T.named_parameters())[0][-1]
    a_S_ts = list(a_S.named_parameters())[0][-1]
    w_ts = list(w.named_parameters())[0][-1]

    source_test, source_label_test = source_data_generate(b_S,sigma,epsilon,20000,d)
    target_test, target_label_test = target_data_generate(20000,d)

    source_test = source_test.T 
     

    for i in range(num_step):

        f_T = w(target_train)
        loss_source = creterion(a_S(w(source_train)), source_label) 
        loss_meta = creterion(target_label_train.T.mm(torch.inverse(f_T.T.mm(f_T) + weight_decay * torch.eye(2).cuda() ).mm(f_T.T).T).mm(f_T.T).T, target_label_train) 

        optimizer.zero_grad()
        optimizer1.zero_grad()
        loss = loss_source + 1.0 * loss_meta 
        loss.backward()
        optimizer.step()
        optimizer1.step()
        
        f_T_test = w(target_test)

        if i % display_interval == 0:
            if display_norm and not display_test_loss:
                print('iter: %6d | a_T: %.3f | a_S: %.3f | w: %.3f'%(
                                                 i,
                                                 torch.norm(a_T_ts).data.cpu().numpy(), 
                                                 torch.norm(a_S_ts).data.cpu().numpy(),
                                                 torch.norm(w_ts).data.cpu().numpy()))

            if not display_norm and display_test_loss:
                print('iter: %6d | test_loss: %.6f | norm_of_difference: %.6f'%(
                                                 i,
                                                 creterion(target_label_train.T.mm(torch.inverse(f_T.T.mm(f_T) + weight_decay
                * torch.eye(2).cuda()).mm(f_T.T).T).mm(f_T_test.T).T, target_label_test).data.cpu().numpy(),
                                                 torch.norm(target_label_train.T.mm(torch.inverse(f_T.T.mm(f_T) + weight_decay
                * torch.eye(2).cuda()).mm(f_T.T).T).mm(list(w.named_parameters())[0][-1]) - b_T).data.cpu().numpy()))
            
            if display_norm and display_test_loss:
                print('iter: %6d | a_T: %.3f | a_S: %.3f | w: %.3f | test_loss: %.6f | norm_of_difference: %.6f'%(
                                                 i,
                                                 torch.norm(a_T_ts).data.cpu().numpy(), 
                                                 torch.norm(a_S_ts).data.cpu().numpy(),
                                                 torch.norm(w_ts).data.cpu().numpy(),
                                                 creterion(target_label_train.T.mm(torch.inverse(f_T.T.mm(f_T) + weight_decay
                * torch.eye(2).cuda()).mm(f_T.T).T).mm(f_T_test.T).T, target_label_test).data.cpu().numpy(),
                                                 torch.norm(target_label_train.T.mm(torch.inverse(f_T.T.mm(f_T) + weight_decay
                * torch.eye(2).cuda()).mm(f_T.T).T).mm(list(w.named_parameters())[0][-1]) - b_T).data.cpu().numpy()))
        
    print('training loss:     ',loss_source.data.cpu().numpy(),loss_meta.data.cpu().numpy())

    f_T_test = w(target_test)

    loss_source_test = creterion(a_S(w(source_test)), source_label_test) 
    loss_target_test = creterion(target_label_train.T.mm(torch.inverse(f_T.T.mm(f_T) + weight_decay
                * torch.eye(2).cuda()).mm(f_T.T).T).mm(f_T_test.T).T, target_label_test)
    print('test loss:         ',loss_source_test.data.cpu().numpy(), loss_target_test.data.cpu().numpy())

    dev_tgt = torch.norm(target_label_train.T.mm(torch.inverse(f_T.T.mm(f_T) + weight_decay
                * torch.eye(2).cuda()).mm(f_T.T).T).mm(list(w.named_parameters())[0][-1]) - b_T)
    dev_src = torch.norm(list(a_S.named_parameters())[0][-1].mm(list(w.named_parameters())[0][-1]) - b_S)
    print('norm of difference: ',dev_src.data.cpu().numpy(),dev_tgt.data.cpu().numpy())

def source(sigma,num_source,num_target,d,lr,weight_decay,num_step,display_norm,display_test_loss,display_interval):
    
     
    source_train, source_label = source_data_generate(b_S,sigma,epsilon,num_source,d)
    target_train, target_label_train = target_data_generate(num_target,d)
    w = nn.Linear(500,2,bias=False).cuda() 
    a_S = nn.Linear(2,2,bias=False).cuda() 
    a_T = nn.Linear(2,2,bias=False).cuda() 

    def weights_init(m):                                                                
        nn.init.normal_(m.weight.data, 0.0, 0.01)                                                  
        #nn.init.constant_(m.weight.data, 0)                          

    w.apply(weights_init)
    a_S.apply(weights_init) 
    a_T.apply(weights_init) 

    optimizer = torch.optim.SGD(w.parameters(),lr=lr,weight_decay=weight_decay,momentum=0)
    optimizer1 = torch.optim.SGD(a_S.parameters(),lr=lr,weight_decay=weight_decay,momentum=0)
   
    creterion = torch.nn.CrossEntropyLoss()

    source_train = source_train.T 
    

    a_T_ts = list(a_T.named_parameters())[0][-1]
    a_S_ts = list(a_S.named_parameters())[0][-1]
    w_ts = list(w.named_parameters())[0][-1]

    source_test, source_label_test = source_data_generate(b_S,sigma,epsilon,20000,d)
    target_test, target_label_test = target_data_generate(20000,d)

    source_test = source_test.T 
     

    for i in range(num_step):
        loss_source = creterion(a_S(w(source_train)), source_label) 

        optimizer.zero_grad()
        optimizer1.zero_grad()
        loss = loss_source
        loss.backward()
        optimizer.step()
        optimizer1.step()
        
        f_T = w(target_train)
        f_T_test = w(target_test)

        if i % display_interval == 0:
            if display_norm and not display_test_loss:
                print('iter: %6d | a_T: %.3f | a_S: %.3f | w: %.3f'%(
                                                 i,
                                                 torch.norm(a_T_ts).data.cpu().numpy(), 
                                                 torch.norm(a_S_ts).data.cpu().numpy(),
                                                 torch.norm(w_ts).data.cpu().numpy()))

            if not display_norm and display_test_loss:
                print('iter: %6d | test_loss: %.6f | norm_of_difference: %.6f'%(
                                                 i,
                                                 creterion(target_label_train.T.mm(torch.inverse(f_T.T.mm(f_T) + weight_decay
                * torch.eye(2).cuda()).mm(f_T.T).T).mm(f_T_test.T).T, target_label_test).data.cpu().numpy(),
                                                 torch.norm(target_label_train.T.mm(torch.inverse(f_T.T.mm(f_T) + weight_decay
                * torch.eye(2).cuda()).mm(f_T.T).T).mm(list(w.named_parameters())[0][-1]) - b_T).data.cpu().numpy()))
            
            if display_norm and display_test_loss:
                print('iter: %6d | a_T: %.3f | a_S: %.3f | w: %.3f | test_loss: %.6f | norm_of_difference: %.6f'%(
                                                 i,
                                                 torch.norm(a_T_ts).data.cpu().numpy(), 
                                                 torch.norm(a_S_ts).data.cpu().numpy(),
                                                 torch.norm(w_ts).data.cpu().numpy(),
                                                 creterion(target_label_train.T.mm(torch.inverse(f_T.T.mm(f_T) + weight_decay
                * torch.eye(2).cuda()).mm(f_T.T).T).mm(f_T_test.T).T, target_label_test).data.cpu().numpy(),
                                                 torch.norm(target_label_train.T.mm(torch.inverse(f_T.T.mm(f_T) + weight_decay
                * torch.eye(2).cuda()).mm(f_T.T).T).mm(list(w.named_parameters())[0][-1]) - b_T).data.cpu().numpy()))

    print('training loss:     ',loss_source.data.cpu().numpy())

    f_T = w(target_train)
    f_T_test = w(target_test)

    loss_source_test = creterion(a_S(w(source_test)), source_label_test) 
    loss_target_test = creterion(target_label_train.T.mm(torch.inverse(f_T.T.mm(f_T) + weight_decay
                * torch.eye(2).cuda()).mm(f_T.T).T).mm(f_T_test.T).T, target_label_test)

    print('test loss:         ',loss_target_test.data.cpu().numpy())
    dev_tgt = torch.norm(target_label_train.T.mm(torch.inverse(f_T.T.mm(f_T) + weight_decay
                * torch.eye(2).cuda()).mm(f_T.T).T).mm(list(w.named_parameters())[0][-1]) - b_T)

    dev_src = torch.norm(list(a_S.named_parameters())[0][-1].mm(list(w.named_parameters())[0][-1]) - b_S)
    print('norm of difference:',dev_tgt.data.cpu().numpy())

def joint_training_fine_tuning(sigma,num_source,num_target,d,lr,weight_decay,num_step,display_norm,display_test_loss,display_interval):
    
    # generate data
     
    source_train, source_label = source_data_generate(b_S,sigma,num_source,d)
    target_train, target_label_train = target_data_generate(num_target,d)
    
    # set the model
    # w 2*500
    # a_S and a_T 1*2 
    w = nn.Linear(500,2,bias=False).cuda()
    a_S = nn.Linear(2,2,bias=False).cuda()
    a_T = nn.Linear(2,2,bias=False).cuda()

    # init the model
    # you can choose gaussian or constant
    def weights_init(m):                                                                
        nn.init.normal_(m.weight.data, 0.0, 0.01)                                                  
        #nn.init.constant_(m.weight.data, 0)                          

    w.apply(weights_init)
    a_S.apply(weights_init) 
    a_T.apply(weights_init) 


    # set the optimizer 
    optimizer = torch.optim.SGD(w.parameters(),lr=lr,weight_decay=weight_decay,momentum=0)
    optimizer1 = torch.optim.SGD(a_S.parameters(),lr=lr,weight_decay=weight_decay,momentum=0)
    optimizer2 = torch.optim.SGD(a_T.parameters(),lr=lr,weight_decay=weight_decay,momentum=0)

    # using mse loss
    creterion = torch.nn.CrossEntropyLoss()

    # set the data for training
    source_train = source_train.T
    target_train = target_train.T


    a_T_ts = list(a_T.named_parameters())[0][-1]
    a_S_ts = list(a_S.named_parameters())[0][-1]
    w_ts = list(w.named_parameters())[0][-1]


    # generate test data
    source_test, source_label_test = source_data_generate(b_S,sigma,epsilon,20000,d)
    target_test, target_label_test = target_data_generate(20000,d)

    source_test = source_test.T
    

    # train the model
    for i in range(num_step):
        loss_source = creterion(a_S(w(source_train)), source_label) 
        loss_target = creterion(a_T(F.relu(w(target_train))), target_label_train) 

        optimizer.zero_grad()
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        loss = loss_source + loss_target
        loss.backward()
        optimizer.step()
        optimizer1.step()
        optimizer2.step()
        
        if i % display_interval == 0:
            if display_norm and not display_test_loss:
                print('iter: %6d | a_T: %.3f | a_S: %.3f | w: %.3f'%(
                                                 i,
                                                 torch.norm(a_T_ts).data.cpu().numpy(), 
                                                 torch.norm(a_S_ts).data.cpu().numpy(),
                                                 torch.norm(w_ts).data.cpu().numpy()))

            if not display_norm and display_test_loss:
                print('iter: %6d | test_loss: %.6f | norm_of_difference: %.6f'%(
                                                 i,
                                                 torch.sum(torch.max(a_T(F.relu(w(target_test))), dim = -1)[-1] == target_label_test).data.cpu().numpy(),
                                                 torch.norm(list(a_T.named_parameters())[0][-1].mm(list(w.named_parameters())[0][-1]) - b_T).data.cpu().numpy()))
            
            if display_norm and display_test_loss:
                print('iter: %6d | a_T: %.3f | a_S: %.3f | w: %.3f | test_loss: %.6f | norm_of_difference: %.6f'%(
                                                 i,
                                                 torch.norm(a_T_ts).data.cpu().numpy(), 
                                                 torch.norm(a_S_ts).data.cpu().numpy(),
                                                 torch.norm(w_ts).data.cpu().numpy(),
                                                 torch.sum(torch.max(a_T(F.relu(w(target_test))), dim = -1)[-1] == target_label_test).data.cpu().numpy(),
                                                 torch.norm(list(a_T.named_parameters())[0][-1].mm(list(w.named_parameters())[0][-1]) - b_T).data.cpu().numpy()))


    # print training loss     
    print('training loss:     ',loss_source.data.cpu().numpy(),loss_target.data.cpu().numpy())

    # calculate test loss 
    loss_source_test = creterion(a_S(w(source_test)), source_label_test) 
    loss_target_test = creterion(a_T(F.relu(w(target_test))), target_label_test)
    print('test loss:         ',loss_source_test.data.cpu().numpy(), loss_target_test.data.cpu().numpy())


    # calculate \|aw - b\|
    norm_tgt = torch.norm(list(a_T.named_parameters())[0][-1].mm(list(w.named_parameters())[0][-1]) - b_T)
    norm_src = torch.norm(list(a_S.named_parameters())[0][-1].mm(list(w.named_parameters())[0][-1]) - b_S)
    print('norm of difference:',norm_src.data.cpu().numpy(),norm_tgt.data.cpu().numpy())
    

    for i in range(num_step):
        loss_target = creterion(a_T(F.relu(w(target_train))), target_label_train) 

        #optimizer.zero_grad()
        optimizer2.zero_grad()
        loss = loss_target
        loss.backward()
        #optimizer.step()
        optimizer2.step()
        
        if i % display_interval == 0:
            if display_norm and not display_test_loss:
                print('iter: %6d | a_T: %.3f | a_S: %.3f | w: %.3f'%(
                                                 i,
                                                 torch.norm(a_T_ts).data.cpu().numpy(), 
                                                 torch.norm(a_S_ts).data.cpu().numpy(),
                                                 torch.norm(w_ts).data.cpu().numpy()))

            if not display_norm and display_test_loss:
                print('iter: %6d | test_loss: %.6f | norm_of_difference: %.6f'%(
                                                 i,
                                                 creterion(a_T(F.relu(w(target_test))), target_label_test),
                                                 torch.norm(list(a_T.named_parameters())[0][-1].mm(list(w.named_parameters())[0][-1]) - b_T).data.cpu().numpy()))
            
            if display_norm and display_test_loss:
                print('iter: %6d | a_T: %.3f | a_S: %.3f | w: %.3f | test_loss: %.6f | norm_of_difference: %.6f'%(
                                                 i,
                                                 torch.norm(a_T_ts).data.cpu().numpy(), 
                                                 torch.norm(a_S_ts).data.cpu().numpy(),
                                                 torch.norm(w_ts).data.cpu().numpy(),
                                                 torch.sum(torch.max(a_T(F.relu(w(target_test))), dim = -1)[-1] == target_label_test).data.cpu().numpy(),
                                                 torch.norm(list(a_T.named_parameters())[0][-1].mm(list(w.named_parameters())[0][-1]) - b_T).data.cpu().numpy()))



    # print training loss     
    print('training loss_ft:  ',loss_source.data.cpu().numpy(),loss_target.data.cpu().numpy())


    # calculate test loss 
    loss_source_test = creterion(a_S(w(source_test)), source_label_test) 
    loss_target_test = creterion(a_T(F.relu(w(target_test))), target_label_test)
    print('test loss:         ',loss_source_test.data.cpu().numpy(), loss_target_test.data.cpu().numpy())


    # calculate \|aw - b\|
    norm_tgt = torch.norm(list(a_T.named_parameters())[0][-1].mm(list(w.named_parameters())[0][-1]) - b_T)
    norm_src = torch.norm(list(a_S.named_parameters())[0][-1].mm(list(w.named_parameters())[0][-1]) - b_S)
    print('norm of difference:',norm_src.data.cpu().numpy(),norm_tgt.data.cpu().numpy())

def fine_tune(delta,sigma,beta,alpha,num_source,num_target,d,lr,weight_decay,num_step,display_norm,display_test_loss,display_interval):
    
     
    target_train, target_label_train = target_data_generate(num_target,d)
    
    w = nn.Linear(500,2,bias=False).cuda() 
    a_S = nn.Linear(2,2,bias=False).cuda() 
    a_T = nn.Linear(2,2,bias=False).cuda() 
    
    def weights_init(m):                                                                
        #nn.init.normal_(m.weight.data, 0.0, 0.01)                                                  
        nn.init.constant_(m.weight.data, 0)                          
    
    w.apply(weights_init)
    a_S.apply(weights_init) 
    a_T.apply(weights_init) 
    w.linear.data[0] += 1
    
    optimizer = torch.optim.SGD(w.parameters(),lr=lr,weight_decay=weight_decay,momentum=0)
    optimizer1 = torch.optim.SGD(a_S.parameters(),lr=lr,weight_decay=weight_decay,momentum=0)
    optimizer2 = torch.optim.SGD(a_T.parameters(),lr=lr,weight_decay=weight_decay,momentum=0)

    creterion = torch.nn.CrossEntropyLoss()
 
    

    a_T_ts = list(a_T.named_parameters())[0][-1]
    a_S_ts = list(a_S.named_parameters())[0][-1]
    w_ts = list(w.named_parameters())[0][-1]


    target_test, target_label_test = target_data_generate(20000,d)
    
     

    for i in range(num_step):
        loss_target = creterion(a_T(F.relu(w(target_train))), target_label_train) 

        optimizer.zero_grad()
        optimizer2.zero_grad()
        loss = loss_target
        loss.backward()
        optimizer.step()
        optimizer2.step()
        
        if i % display_interval == 0:
            if display_norm and not display_test_loss:
                print('iter: %6d | a_T: %.3f | a_S: %.3f | w: %.3f'%(
                                                 i,
                                                 torch.norm(a_T_ts).data.cpu().numpy(), 
                                                 torch.norm(a_S_ts).data.cpu().numpy(),
                                                 torch.norm(w_ts).data.cpu().numpy()))

            if not display_norm and display_test_loss:
                print('iter: %6d | test_loss: %.6f | norm_of_difference: %.6f'%(
                                                 i,
                                                 torch.sum(torch.max(a_T(F.relu(w(target_test))), dim = -1)[-1] == target_label_test).data.cpu().numpy(),
                                                 torch.norm(list(a_T.named_parameters())[0][-1].mm(list(w.named_parameters())[0][-1]) - b_T).data.cpu().numpy()))
            
            if display_norm and display_test_loss:
                print('iter: %6d | a_T: %.3f | a_S: %.3f | w: %.3f | test_loss: %.6f | norm_of_difference: %.6f'%(
                                                 i,
                                                 torch.norm(a_T_ts).data.cpu().numpy(), 
                                                 torch.norm(a_S_ts).data.cpu().numpy(),
                                                 torch.norm(w_ts).data.cpu().numpy(),
                                                 torch.sum(torch.max(a_T(F.relu(w(target_test))), dim = -1)[-1] == target_label_test).data.cpu().numpy(),
                                                 torch.norm(list(a_T.named_parameters())[0][-1].mm(list(w.named_parameters())[0][-1]) - b_T).data.cpu().numpy()))


    print('training loss:     ',loss_target.data.cpu().numpy())

    loss_target_test = creterion(a_T(F.relu(w(target_test))), target_label_test)
    print('test loss:         ',loss_target_test.data.cpu().numpy())

    

for alg in algorithm:
    if alg == 'joint-training':
        print(' ')
        print('joint-training')
        joint_training(delta = delta,
                       sigma = sigma,
                       epsilon = epsilon,
                       num_source = num_source,
                       num_target = num_target,
                       d = d,
                       lr = lr,
                       weight_decay = weight_decay,
                       num_step = num_step,
                       display_norm = display_norm,
                       display_test_loss = display_test_loss,
                       display_interval = display_interval)

    if alg == 'joint-training+fine-tuning':
        print(' ')
        print('joint-training+fine-tuning')
        joint_training_fine_tuning(delta = delta,
                       sigma = sigma,
                       epsilon = epsilon,
                       num_source = num_source,
                       num_target = num_target,
                       d = d,
                       lr = lr,
                       weight_decay = weight_decay,
                       num_step = num_step,
                       display_norm = display_norm,
                       display_test_loss = display_test_loss,
                       display_interval = display_interval)

    if alg == 'meta-learning':
        print(' ')
        print('meta-learning')
        meta_learning(delta = delta,
                      sigma = sigma,
                      epsilon = epsilon,
                      num_source = num_source,
                      num_target = num_target,
                      d = d,
                      lr = lr,
                      weight_decay = weight_decay,
                      num_step = num_step,
                      display_norm = display_norm,
                      display_test_loss = display_test_loss,
                      display_interval = display_interval)
    if alg == 'meta-learning-shuffle':
        print(' ')
        print('meta-learning-shuffle')
        meta_learning_shuffle(delta = delta,
                      sigma = sigma,
                      epsilon = epsilon,
                      num_source = num_source,
                      num_target = num_target,
                      d = d,
                      lr = lr,
                      weight_decay = weight_decay,
                      num_step = num_step,
                      display_norm = display_norm,
                      display_test_loss = display_test_loss,
                      display_interval = display_interval)

    if alg == 'algorithm1':
        print(' ')
        print('algorithm1')
        algorithm1(delta = delta,
                   sigma = sigma,
                   epsilon = epsilon,
                   num_source = num_source,
                   num_target = num_target,
                   d = d,
                   lr = lr,
                   weight_decay = weight_decay,
                   num_step = num_step,
                   display_norm = display_norm,
                   display_test_loss = display_test_loss,
                   display_interval = display_interval)

    if alg == 'source':
        print(' ')
        print('source')
        source(delta = delta,
               sigma = sigma,
               epsilon = epsilon,
               num_source = num_source,
               num_target = num_target,
               d = d,
               lr = lr,
               weight_decay = weight_decay,
               num_step = num_step,
               display_norm = display_norm,
               display_test_loss = display_test_loss,
               display_interval = display_interval)

    if alg == 'target':
        print(' ')
        print('target')
        target(delta = delta,
               sigma = sigma,
               epsilon = epsilon,
               num_source = num_source,
               num_target = num_target,
               d = d,
               lr = lr,
               weight_decay = weight_decay,
               num_step = num_step,
               display_norm = display_norm,
               display_test_loss = display_test_loss,
               display_interval = display_interval)
    if alg == 'fine-tune':
        print(' ')
        print('fine-tune')
        fine_tune(delta = delta,
               sigma = sigma,
               epsilon = epsilon,
               num_source = num_source,
               num_target = num_target,
               d = d,
               lr = lr,
               weight_decay = weight_decay,
               num_step = num_step,
               display_norm = display_norm,
               display_test_loss = display_test_loss,
               display_interval = display_interval)