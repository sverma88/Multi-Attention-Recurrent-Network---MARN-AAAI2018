import torch
import time,numpy
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import h5py
import random
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score

from marn_module import MARN
import marn_data_loader as loader

def marn_train(inputy,inputx_v,inputx_a,inputx_t,inputy_test,inputx_v_test,inputx_a_test,inputx_t_test,inputy_valid,inputx_v_valid,inputx_a_valid,inputx_t_valid, configs):

    in_size=[inputx_v.shape[2],inputx_a.shape[2],inputx_t.shape[2]]
    print("dimensionality of features:", in_size)

    batch_size = configs["batch_size"]

    print("len of train features:", inputy.shape[0])
    print("len of test features:", inputy_test.shape[0])
    print("len of valid features:", inputy_valid.shape[0])

    len_mem = [configs["len_mem_v"], configs["len_mem_a"], configs["len_mem_t"]]
    len_z_m = [configs["len_z_m_v"], configs["len_z_m_a"], configs["len_z_m_t"]]
    len_z_hidden = configs["len_z_hidden"]
    K = configs["K"]

    print("len_mem:",len_mem)
    print("len_z_m:",len_z_m)
    print("len_z_hidden:",len_z_hidden)
    print("K:",K)

    #Build the model
    marn_model = MARN(in_size, len_mem, len_z_m, len_z_hidden, K)

    criterion = nn.L1Loss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    marn_model = marn_model.to(device)
    optimizer = torch.optim.Adam(marn_model.parameters(),lr=configs["lr"])

    criterion = criterion.to(device)
    scheduler = ReduceLROnPlateau(optimizer,mode='min',patience=100,factor=0.5,verbose=True)



    ####### Function to perfrom training of MARN
    def train(marn_model, batch_size, inputy, inputx_v, inputx_a, inputx_t, optimizer, criterion):
        epoch_loss = 0
        marn_model.train()
        total_n = inputx_a.shape[0]
        num_batches = total_n // batch_size

        for batch in range(num_batches):
            start = batch * batch_size
            end = (batch+1) * batch_size
            optimizer.zero_grad()

            vid_range = [start, end]
            batch_y = inputy[start:end]

            marn_model.zero_grad()
            y_hat = marn_model.forward(batch_size, inputx_v, inputx_a, inputx_t, vid_range).reshape(-1)


            train_loss = criterion(y_hat, batch_y)
            train_loss.backward()
            optimizer.step()
            epoch_loss += train_loss.item()

        return epoch_loss / num_batches


    ####### Function to perfrom validation of MARN
    def evaluate(marn_model, batch_size, inputy_valid, inputx_v_valid, inputx_a_valid, inputx_t_valid, criterion):

        epoch_loss = 0
        marn_model.eval()
        total_n = inputx_a_valid.shape[0]
        num_batches = total_n // batch_size

        for batch in range(num_batches):
            start = batch * batch_size
            end = (batch+1) * batch_size
            optimizer.zero_grad()

            vid_range = [start, end]
            batch_y = inputy_valid[start:end]

            marn_model.zero_grad()
            y_hat = marn_model.forward(batch_size, inputx_v_valid, inputx_a_valid, inputx_t_valid, vid_range).reshape(-1)

            train_loss = criterion(y_hat, batch_y)
            train_loss.backward()
            optimizer.step()
            epoch_loss += train_loss.item()

        return epoch_loss / num_batches

    ####### Function to get predictions n from  MARN
    def predict(marn_model, batch_size, inputx_v, inputx_a, inputx_t):
        marn_model.eval()
        total_n = inputx_a.shape[0]
        num_batches = total_n // batch_size
        predictions = []

        with torch.no_grad():
            for batch in range(num_batches + 1):
                start = batch * batch_size

                if (total_n - start) < batch_size:
                    end = total_n
                    batch_size = end - start

                else:
                    end = (batch + 1) * batch_size

                vid_range = [start, end]

                y_hat = marn_model.forward(batch_size, inputx_v, inputx_a, inputx_t, vid_range).reshape(-1)
                predictions.append(y_hat)

                if end == total_n:
                    break

        return torch.cat(predictions, dim=0)

    #Train the model
    best_valid = float('inf')
    for epoch in range(configs["max_epoch"]):
        print("Current Epoch ------>",epoch)

        train_loss = train(marn_model,batch_size, inputy, inputx_v, inputx_a, inputx_t, optimizer, criterion)
        valid_loss = evaluate(marn_model, batch_size, inputy, inputx_v_valid, inputx_a_valid, inputx_t_valid, criterion)
        scheduler.step(valid_loss)

        ###### Condition to save the model
        if valid_loss <= best_valid:
            best_valid = valid_loss
            print("Epoch:{} \t train_loss:{} \t valid_loss: {} \n saving the model...".format(epoch,train_loss,valid_loss))
            best_model_path = loader.save_model("epoch_" + str(epoch), marn_model)

        else:
            print ("Epoch:{} \t train_loss:{} \t valid_loss:{}".format(epoch,train_loss,valid_loss))

    # epochs exhausted
    print("epoch_done")
    marn_model=loader.load_model(best_model_path)
    predictions = predict(marn_model, batch_size, inputx_v_test, inputx_a_test, inputx_t_test)

    print("Type of predictions:",type(predictions))
    print("Type of inputy_test:",type(inputy_test))

    predictions = predictions.cpu().detach().numpy()
    inputy_test = inputy_test.cpu().detach().numpy()

    mae = np.mean(np.absolute(predictions-inputy_test))
    corr = np.corrcoef(predictions,inputy_test)[0][1]
    mult_acc = round(sum(np.round(predictions) == np.round(inputy_test))/float(len(inputy_test)),5)
    true_label = (inputy_test >= 0)
    predicted_label = (predictions >= 0)
    f1 = round(f1_score(true_label,predicted_label,average='weighted'),5)
    binary_acc = accuracy_score(true_label, predicted_label)

    print("Mae: {} \n Corr : {} \n binary_acc: {} \n f1:{} \n mult_acc:{} \n" .format(mae,corr,binary_acc,f1,mult_acc))

if __name__ == "__main__":

    ### read the dataset

    dataset = "mosei"  ### Dataset name, can be mosei or mosi
    train_y, train_v, train_a, train_t, test_y, test_v, test_a, test_t, valid_y, valid_v, \
                                                    valid_a, valid_t = loader.load(dataset)


    #hyperparameters for LSTM Hybrid
    configs = dict()

    configs["len_mem_v"] = 32
    configs["len_mem_a"] = 30
    configs["len_mem_t"] = 128

    configs["len_z_m_v"] = 16
    configs["len_z_m_a"] = 16
    configs["len_z_m_t"] = 32

    configs["len_z_hidden"] = 10
    configs["K"] = 4  ### The number of attentions for MAB Block

    configs["lr"] = 0.0001
    configs["batch_size"] = 256

    configs["max_epoch"] = 250
    configs["train_display_freq"] = 1000
    configs["test_display_freq"] = 1000

    #the main function
    marn_train(train_y, train_v, train_a, train_t, test_y, test_v, test_a, test_t, valid_y, valid_v, valid_a, valid_t, configs)


