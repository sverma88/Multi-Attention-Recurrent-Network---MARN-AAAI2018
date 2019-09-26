import torch
import time,numpy
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import h5py
import pickle,sys,time


class LSTHM(nn.Module):
    '''
    Copied From CMU SDK mmsdk/mmmodelsdk
    LSTMH Hybrid Class as in paper Multi-attention Recurrent Network - AAAI 2018
    '''
    def __init__(self, cell_size, in_size, hybrid_in_size):
        super(LSTHM, self).__init__()
        self.cell_size=cell_size
        self.in_size=in_size
        self.W=nn.Linear(in_size,4*self.cell_size)
        self.U=nn.Linear(cell_size,4*self.cell_size)
        self.V=nn.Linear(hybrid_in_size,4*self.cell_size)
        
    def __call__(self, x, ctm1, htm1, ztm1):
        return self.forward(x,ctm1,htm1,ztm1)
    
    def forward(self, x, ctm1, htm1, ztm1):
        input_affine = self.W(x)
        output_affine = self.U(htm1)
        hybrid_affine = self.V(ztm1)
        
        sums = input_affine + output_affine + hybrid_affine

        #biases are already part of W and U and V
        f_t=torch.sigmoid(sums[:, :self.cell_size])
        i_t=torch.sigmoid(sums[:, self.cell_size:2*self.cell_size])
        o_t=torch.sigmoid(sums[:, 2*self.cell_size:3*self.cell_size])
        ch_t=torch.tanh(sums[:, 3*self.cell_size:])
        c_t=f_t*ctm1+i_t*ch_t
        h_t=torch.tanh(c_t)*o_t
        return c_t,h_t




class MultipleAttentionFusion(nn.Module):

        '''
    Edited from CMU SDK mmsdk/mmmodelsdk
    Multi Attention Block as in paper Multi-attention Recurrent Network - AAAI 2018
    '''

    def __init__(self,len_mem,len_z_m,num_atts):
        super(MultipleAttentionFusion, self).__init__()

        self.num_atts=num_atts
        self.attention_model =nn.Sequential(nn.Linear(sum(len_mem),sum(len_mem)*num_atts))
        self.small_netv =nn.Sequential(nn.Linear(len_mem[0]*num_atts,len_z_m[0]))
        self.small_neta =nn.Sequential(nn.Linear(len_mem[1]*num_atts,len_z_m[1]))
        self.small_nett =nn.Sequential(nn.Linear(len_mem[2]*num_atts,len_z_m[2]))
        self.dim_reduce_nets=[self.small_netv,self.small_neta,self.small_nett]

    def __call__(self,in_modalities):
        return self.forward(in_modalities)


    def forward(self,in_modalities):
		#getting some simple integers out
		num_modalities=len(in_modalities)
		#simply the tensor that goes into attention_model
		in_tensor=torch.cat(in_modalities,dim=1)
		#calculating attentions
		atts=F.softmax(self.attention_model(in_tensor),dim=1)
		#calculating the tensor that will be multiplied with the attention
		out_tensor=torch.cat([in_modalities[i].repeat(1,self.num_atts) for i in range(num_modalities)],dim=1)
		#calculating the attention
		att_out=atts*out_tensor

		#now to apply the dim_reduce networks
		#first back to however modalities were in the problem
		start=0
		out_modalities=[]
		for i in range(num_modalities):
			modality_length=in_modalities[i].shape[1]*self.num_atts
			out_modalities.append(att_out[:,start:start+modality_length])
			start=start+modality_length
	
		#apply the dim_reduce
		dim_reduced=[self.dim_reduce_nets[i](out_modalities[i]) for i in range(num_modalities)]
		#multiple attention done :)

		z=torch.cat(dim_reduced,dim=1)
		return z,out_modalities

class Zto1(nn.Module):
    def __init__(self,in_size,hidden_size):
        super(Zto1, self).__init__()
        self.n1=nn.Linear(in_size,hidden_size)
        self.n2=nn.Linear(hidden_size,1)


    def forward(self, input):
        y1=torch.relu(self.n1(input))
        y2=self.n2(y1)
        return y2


class Map_pred(nn.Module):
    def __init__(self,in_size,hidden_size):
        super(Map_pred, self).__init__()
        self.n1 = nn.Linear(in_size,hidden_size)
        self.n2 = nn.Linear(hidden_size,1)


    def forward(self, input):
        y1 = torch.relu(self.n1(input))
        y2 = self.n2(y1)
        return y2



class MARN(nn.Module):
    def __init__(self, in_size, len_mem, len_z_m, z_hidden, K):
        super(MARN, self).__init__()

        #the size of LSTMH of each modality
        self.len_mem=len_mem

        #size of the cross-view dynamics of each modality
        self.len_z_m = len_z_m

        #whole size of the cross-view dynamics
        self.len_z = sum(self.len_z_m)

        #the num of attention coefﬁcients
        self.K = K

        #set the LSTHM_model of each modality
        self.fmodel_v = LSTHM(len_mem[0],in_size[0],self.len_z)
        self.fmodel_a = LSTHM(len_mem[1],in_size[1],self.len_z)
        self.fmodel_t = LSTHM(len_mem[2],in_size[2],self.len_z)

        #MAB_model
        self.fmodel = MultipleAttentionFusion(self.len_mem,self.len_z_m,self.K)

        #set the dim-decrease model for z, z_hidden is the size of the hidden layer
        self.pred_model = Map_pred(self.len_z, z_hidden)



    #pass in the init zeros
    def forward(self, batch_size, inputx_v, inputx_a, inputx_t, vid_range):
        start = vid_range[0]
        end = vid_range[1]

        '''
        Setup variables for LSTMH 
        '''
        self.c_v = torch.zeros(batch_size,self.len_mem[0]).cuda()
        self.c_a = torch.zeros(batch_size,self.len_mem[1]).cuda()
        self.c_t = torch.zeros(batch_size,self.len_mem[2]).cuda()

        self.h_v = torch.zeros(batch_size,self.len_mem[0]).cuda()
        self.h_a = torch.zeros(batch_size,self.len_mem[1]).cuda()
        self.h_t = torch.zeros(batch_size,self.len_mem[2]).cuda()

        self.z = torch.zeros(batch_size,self.len_z).cuda()
        
        # loop for the whole seq
        for t in range(0,inputx_v.shape[1]):

            # processing each modality with a LSTM Hybrid
            c_v,h_v = self.fmodel_v(inputx_v[start:end, t, :],self.c_v,self.h_v,self.z)
            c_a,h_a = self.fmodel_a(inputx_a[start:end, t, :],self.c_a,self.h_a,self.z)
            c_t,h_t = self.fmodel_t(inputx_t[start:end, t, :],self.c_t,self.h_t,self.z)

            concate_h = [h_v,h_a,h_t]  # Line 14 of Alg. 1 in the paper

            z,_= self.fmodel(concate_h)  # Apply the MAB block

            ### updating the values for next sequence of same utterance
            self.c_v = c_v
            self.c_a = c_a
            self.c_t = c_t

            self.h_t = h_t
            self.h_t = h_t
            self.h_t = h_t

            self.z = z

        ## The predicted sentiment for a single utterance
        y_hat = self.pred_model(z)


        return y_hat
