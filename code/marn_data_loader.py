import torch
import time,numpy
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import h5py 
import pickle,sys,time,os

def save_model(filename, model):
    path_base = sys.path[0].replace("\\","/")
    path_base = path_base + "/models/"

    timestamp = time.strftime("%Y,%m,%d,%H,%M,%S")

    path = path_base + timestamp + "_" + filename + "_model.p"
    torch.save(model, path)

    return path

def load_model(filename):
    model = torch.load(filename)
    return model

def save_log(filename, log):
    path_base = sys.path[0].replace("\\","/")
    path_base = path_base + "/logs/"

    with open(path_base + filename,'a') as f:
        f.write(log)

def save_list(list):
    path_base = sys.path[0].replace("\\","/")
    path_base = path_base + "/preds/"
    timestamp = time.strftime("%Y,%m,%d,%H,%M,%S")

    with open(path_base + timestamp + "_list.txt",'w') as f:
        for i in range(len(list)):
            f.write(str(list[i].item())+"\n")
    

def load(data):

    path_base = sys.path[0].replace("\\","/")
    path_base = path_base[0:int(path_base.rfind('/'))]

    if data=="mosi":
        path_base = path_base + "/data/mosi/"
        path_x = path_base+'X_train.h5'
        path_y = path_base+'y_train.h5'
        path_x_test = path_base+'X_test.h5'
        path_y_test = path_base+'y_test.h5'
        path_x_val = path_base+'X_valid.h5'
        path_y_val = path_base+'y_valid.h5'

        with h5py.File(path_x,'r') as h5:
            content = h5['data']
            print('\n',path_x,'\n',content,'\n')    
            x = np.ones(content.shape)
            content.read_direct(x)
            # print(x)

        with h5py.File(path_y,'r') as h5:
            content = h5['data']
            print('\n',path_y,'\n',content,'\n')    
            y = np.ones(content.shape)
            content.read_direct(y)
            # print(a)

        #train_X
        inputx=Variable(torch.Tensor(x),requires_grad=True).cuda()

        #train_y
        train_y=Variable(torch.Tensor(y),requires_grad=True).cuda()

        train_v = inputx[:,:,305:325]
        train_a = inputx[:,:,300:305]
        train_t = inputx[:,:,:300]

        ##***test***
        with h5py.File(path_x_test,'r') as h5:
            content = h5['data']
            print('\n',path_x_test,'\n',content,'\n')    
            x_test = np.ones(content.shape)
            content.read_direct(x_test)

        with h5py.File(path_y_test,'r') as h5:
            content = h5['data']
            print('\n',path_y_test,'\n',content,'\n')    
            y_test = np.ones(content.shape)
            content.read_direct(y_test)

        #test_X
        inputx_test=Variable(torch.Tensor(x_test),requires_grad=True).cuda()

        #test_y
        test_y=Variable(torch.Tensor(y_test),requires_grad=True).cuda()

        test_v = inputx_test[:,:,305:325]
        test_a = inputx_test[:,:,300:305]
        test_t = inputx_test[:,:,:300]


        ##***validation***
        with h5py.File(path_x_val,'r') as h5:
            content = h5['data']
            print('\n',path_x_val,'\n',content,'\n')
            x_test = np.ones(content.shape)
            content.read_direct(x_test)

        with h5py.File(path_y_test,'r') as h5:
            content = h5['data']
            print('\n',path_y_test,'\n',content,'\n')
            y_test = np.ones(content.shape)
            content.read_direct(y_test)

        #val_X
        inputx_test=Variable(torch.Tensor(x_test),requires_grad=True).cuda()

        #test_y
        val_y=Variable(torch.Tensor(y_test),requires_grad=True).cuda()

        val_v = inputx_test[:,:,305:325]
        val_a = inputx_test[:,:,300:305]
        val_t = inputx_test[:,:,:300]


    if data=="mosei":
        #path
        path_base = path_base + "/data/mosei/"

        path_a_train = path_base+'audio_train.h5'
        path_v_train = path_base+'video_train.h5'
        path_t_train = path_base+'text_train_emb.h5'
        path_y_train = path_base+'y_train.h5'

        path_a_test = path_base+'audio_test.h5'
        path_v_test = path_base+'video_test.h5'
        path_t_test = path_base+'text_test_emb.h5'
        path_y_test = path_base+'y_test.h5'

        path_a_valid = path_base+'audio_valid.h5'
        path_v_valid = path_base+'video_valid.h5'
        path_t_valid = path_base+'text_valid_emb.h5'
        path_y_valid = path_base+'y_valid.h5'

        #load the H5 files
        #****************   train   *****************
        with h5py.File(path_a_train,'r') as h5:
            content = h5['d1'] 
            a_train = np.ones(content.shape)
            content.read_direct(a_train)
        with h5py.File(path_v_train,'r') as h5:
            content = h5['d1'] 
            v_train = np.ones(content.shape)
            content.read_direct(v_train)
        with h5py.File(path_t_train,'r') as h5:
            content = h5['d1'] 
            t_train = np.ones(content.shape)
            content.read_direct(t_train)
        with h5py.File(path_y_train,'r') as h5:
            content = h5['d1']  
            y_train = np.ones(content.shape)
            content.read_direct(y_train)


        # Remove possible nan values
        v_train[v_train != v_train] = 0
        a_train[a_train != a_train] = 0
        t_train[t_train != t_train] = 0

        # normalization the data

        v_max = np.max(np.max(np.abs(v_train), axis=0), axis=0)
        v_max[v_max == 0] = 1

        a_max = np.max(np.max(np.abs(a_train), axis=0), axis=0)
        a_max[a_max == 0] = 1

        a_train = a_train / a_max
        v_train = v_train / v_max


        train_y=Variable(torch.Tensor(y_train),requires_grad=True).cuda()
        train_v=Variable(torch.Tensor(v_train),requires_grad=True).cuda()
        train_a=Variable(torch.Tensor(a_train),requires_grad=True).cuda()
        train_t=Variable(torch.Tensor(t_train),requires_grad=True).cuda()

        #****************   test   *****************
        with h5py.File(path_a_test,'r') as h5:
            content = h5['d1'] 
            a_test = np.ones(content.shape)
            content.read_direct(a_test)
        with h5py.File(path_v_test,'r') as h5:
            content = h5['d1'] 
            v_test = np.ones(content.shape)
            content.read_direct(v_test)
        with h5py.File(path_t_test,'r') as h5:
            content = h5['d1'] 
            t_test = np.ones(content.shape)
            content.read_direct(t_test)
        #get the y_test
        with h5py.File(path_y_test,'r') as h5:
            content = h5['d1']  
            y_test = np.ones(content.shape)
            content.read_direct(y_test)

        v_test[v_test != v_test] = 0
        a_test[a_test != a_test] = 0
        t_test[t_test != t_test] = 0

        a_test = a_test / a_max
        v_test = v_test / v_max
        
        test_y = Variable(torch.Tensor(y_test),requires_grad=True).cuda()
        test_v = Variable(torch.Tensor(v_test),requires_grad=True).cuda()
        test_a = Variable(torch.Tensor(a_test),requires_grad=True).cuda()
        test_t = Variable(torch.Tensor(t_test),requires_grad=True).cuda()

        #****************   valid   *****************
        with h5py.File(path_a_valid,'r') as h5:
            content = h5['d1'] 
            a_valid = np.ones(content.shape)
            content.read_direct(a_valid)
        with h5py.File(path_v_valid,'r') as h5:
            content = h5['d1'] 
            v_valid = np.ones(content.shape)
            content.read_direct(v_valid)
        with h5py.File(path_t_valid,'r') as h5:
            content = h5['d1'] 
            t_valid = np.ones(content.shape)
            content.read_direct(t_valid)
        with h5py.File(path_y_valid,'r') as h5:
            content = h5['d1']  
            y_valid = np.ones(content.shape)
            content.read_direct(y_valid)

        v_valid[v_valid != v_valid] = 0
        a_valid[a_valid != a_valid] = 0
        t_valid[t_valid != t_valid] = 0

        a_valid = a_valid / a_max
        v_valid = v_valid / v_max

        val_y = Variable(torch.Tensor(y_valid),requires_grad=True).cuda()
        val_v = Variable(torch.Tensor(v_valid),requires_grad=True).cuda()
        val_a = Variable(torch.Tensor(a_valid),requires_grad=True).cuda()
        val_t = Variable(torch.Tensor(t_valid),requires_grad=True).cuda()

    return train_y, train_v, train_a, train_t, test_y, test_v, test_a, test_t,val_y, val_v, val_a, val_t
