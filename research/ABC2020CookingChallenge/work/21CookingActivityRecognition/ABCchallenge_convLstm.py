# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 21:43:14 2020

@author: Kazuya
"""

import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchvision import datasets, transforms
from torch.autograd import Variable
import math
import glob
import sys
import pickle
import os
import re

args = sys.argv

###Training param
EPOCH_NUM = 1000 #1 epoch = train all data
NUM_ITER = 1 #1 iter = EPOCH_NUM epochs
MAP_SIZE = 0 # Number of feature map in convolution layers
HIDDEN_SIZE = 24	 #Number of hidden layer in LSTM
NUM_CHANNEL = 0 #input dimention, i.e. x,y,z of accelerometer
MIN_SEQ_LEN = 10
KERNEL_SIZE = 10
users_for_train = args[4]
users_for_test = args[5]
network_id = int(args[1]) #1=ConvLSTM, 2=LSTM, 3=ConvLSTMsingle, 4=LSTMsingle
if int(args[2])==0:
    raw_or_feature = "new"
else:
    raw_or_feature = "feature"
if raw_or_feature=="new":
    NUM_CHANNEL = 3
    MAP_SIZE = NUM_CHANNEL*6
else:
    NUM_CHANNEL = 21
    MAP_SIZE = NUM_CHANNEL*6
if network_id == 2 or network_id == 4:
    MAP_SIZE = 0
if network_id==1 or network_id==2:
    NUM_ITER = 1
else:
    NUM_ITER =4    
    
print(raw_or_feature)
print("network_id:{}".format(network_id))
print("Train{} Test{}".format(users_for_train,users_for_test))
###Label
if int(args[3]) == 0:
    is_activity_recognition = True #Activities if true, otherwise recipe
else:
    is_activity_recognition =False
activities = ["Cut", "Peel", "Open", "Take", "Put", "Pour", "Wash", "Add", "Mix", "other"]
recipes = ["sandwich", "fruitsalad", "cereal"]
bodyparts=["left_hip", "left_wrist", "right_arm", "right_wrist"]
if is_activity_recognition:
    NUM_CLASSES = len(activities)
elif not is_activity_recognition:
    NUM_CLASSES = len(recipes)
###Model class
class ConvLSTM(nn.Module):
    def __init__(self, seq_size, map_size, hidden_size, out_size):
        super(ConvLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.conv1 = torch.nn.Conv1d(seq_size, map_size, groups=seq_size, kernel_size=KERNEL_SIZE)
        self.lstm1 = torch.nn.LSTM(map_size, hidden_size)
        self.fc1 = torch.nn.Linear(hidden_size, out_size)
        self.conv2 = torch.nn.Conv1d(seq_size, map_size, groups=seq_size, kernel_size=KERNEL_SIZE)
        self.lstm2 = torch.nn.LSTM(map_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, out_size)
        self.conv3 = torch.nn.Conv1d(seq_size, map_size, groups=seq_size, kernel_size=KERNEL_SIZE)
        self.lstm3 = torch.nn.LSTM(map_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, out_size)
        self.conv4 = torch.nn.Conv1d(seq_size, map_size, groups=seq_size, kernel_size=KERNEL_SIZE)
        self.lstm4 = torch.nn.LSTM(map_size, hidden_size)
        self.fc4 = torch.nn.Linear(hidden_size, out_size)
        self.fc5 = torch.nn.Linear(out_size*4, out_size)
        
    def __call__(self, x1,x2,x3,x4,l1,l2,l3,l4):
        conv1_out = self.conv1(x1)
        l1 = l1 - (KERNEL_SIZE - 1)
        conv1_out = torch.nn.utils.rnn.pack_padded_sequence(conv1_out.permute(2,0,1),l1,enforce_sorted=False)
        _, lstm_out1 = self.lstm1(conv1_out)
        y1 = self.fc1(lstm_out1[0].view(-1, self.hidden_size))

        conv2_out = self.conv2(x2)
        l2 = l2 - (KERNEL_SIZE - 1)
        conv2_out = torch.nn.utils.rnn.pack_padded_sequence(conv2_out.permute(2,0,1),l2,enforce_sorted=False)
        _, lstm_out2 = self.lstm2(conv2_out)
        y2 = self.fc2(lstm_out2[0].view(-1, self.hidden_size))

        conv3_out = self.conv3(x3)
        l3 =  l3 - (KERNEL_SIZE - 1)
        conv3_out = torch.nn.utils.rnn.pack_padded_sequence(conv3_out.permute(2,0,1),l3,enforce_sorted=False)
        _, lstm_out3 = self.lstm3(conv3_out)
        y3 = self.fc3(lstm_out3[0].view(-1, self.hidden_size))

        conv4_out = self.conv4(x4)
        l4 = l4 - (KERNEL_SIZE - 1)
        conv4_out = torch.nn.utils.rnn.pack_padded_sequence(conv4_out.permute(2,0,1),l4,enforce_sorted=False)
        _, lstm_out4 = self.lstm4(conv4_out)
        y4 = self.fc4(lstm_out4[0].view(-1, self.hidden_size))
        y = torch.cat([y1,y2,y3,y4], dim=1)
        out = self.fc5(y)
        return out

class LSTM(nn.Module):
    def __init__(self, seq_size, hidden_size, out_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm1 = torch.nn.LSTM(seq_size,hidden_size)
        self.fc1 = torch.nn.Linear(hidden_size,out_size)
        self.lstm2 = torch.nn.LSTM(seq_size,hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size,out_size)
        self.lstm3 = torch.nn.LSTM(seq_size,hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size,out_size)
        self.lstm4 = torch.nn.LSTM(seq_size,hidden_size)
        self.fc4 = torch.nn.Linear(hidden_size,out_size)
        self.fc5 = torch.nn.Linear(out_size*4,out_size)
        
    def __call__(self,x1,x2,x3,x4):
        _, lstm_out1 = self.lstm1(x1)
        y1 = self.fc1(lstm_out1[0].view(-1, self.hidden_size))
        _, lstm_out2 = self.lstm2(x2)
        y2 = self.fc2(lstm_out2[0].view(-1, self.hidden_size))
        _, lstm_out3 = self.lstm3(x3)
        y3 = self.fc3(lstm_out3[0].view(-1, self.hidden_size))
        _, lstm_out4 = self.lstm4(x4)
        y4 = self.fc4(lstm_out4[0].view(-1, self.hidden_size))
        y = torch.cat([y1,y2,y3,y4], dim=1)
        out = self.fc5(y)
        return out
    
class ConvLSTM_single(nn.Module):
    def __init__(self, seq_size, map_size,  hidden_size, out_size):
        super(ConvLSTM_single, self).__init__()
        self.hidden_size = hidden_size
        self.conv1 = torch.nn.Conv1d(seq_size, map_size, groups=seq_size, kernel_size=KERNEL_SIZE)
 #       self.conv2 = torch.nn.Conv1d(map_size, map_size, groups=seq_size, kernel_size=KERNEL_SIZE)
 #       self.conv3 = torch.nn.Conv1d(map_size, map_size, groups=seq_size, kernel_size=KERNEL_SIZE)
 #       self.conv4 = torch.nn.Conv1d(map_size, map_size, groups=seq_size, kernel_size=KERNEL_SIZE)
        self.lstm1 = torch.nn.LSTM(map_size, hidden_size)
        self.fc1 = torch.nn.Linear(hidden_size, out_size)
        
    def __call__(self, x, length):
        conv1_out = self.conv1(x)
 #       conv2_out = self.conv2(conv1_out)
 #       conv3_out = self.conv3(conv2_out)
 #       conv4_out = self.conv4(conv3_out)
        length = length - (KERNEL_SIZE - 1) * 1
        conv_out = torch.nn.utils.rnn.pack_padded_sequence(conv1_out.permute(2,0,1),length,enforce_sorted=False)
        _, lstm_out1 = self.lstm1(conv_out)
        y = self.fc1(lstm_out1[0].view(-1, self.hidden_size))
        return y

class LSTM_single(nn.Module):
    def __init__(self, seq_size, hidden_size, out_size):
        super(LSTM_single, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(seq_size, hidden_size)
        self.lstm2 = torch.nn.LSTM(hidden_size, hidden_size)
        self.fc1 = torch.nn.Linear(hidden_size, out_size)
        
    def __call__(self, x):
        _, lstm_out = self.lstm(x)
        _, lstm_out = self.lstm2(lstm_out[0])
        y = self.fc1(lstm_out[0].view(-1,self.hidden_size))
        return y

###Functions
def ActToIndex(activity, default=False):#Activity name to activity number
    if activity in activities:
        return activities.index(activity)
    else:
        return default
    
def RecpieToIndex(recipe, default=False):#Recipe name to recipe number
    if recipe in recipes:
        return recipes.index(recipe)
    else:
        return default

def MultiLabelBinarizer(label, NUM_CLASSES):#Multiple activity names to multi-label one-hot vector
    y = np.zeros(NUM_CLASSES)
    for i in label:
        y[i] = 1
    return y.tolist()

def MultiLabelDebinarizer(label):#Multi-label one-hot vector to multiple activity names
    label = label.cpu().numpy()
    output = [[] for i in range(len(label))]
    i, j = np.where(label>0)
    for k in range(len(i)):
        output[i[k]].append(j[k])
    return output

def CalcMultiLabelAccuracy(pred, target):#Calculate accuracy of multi-label output compered with groundtruth
    pred = pred.detach().cpu().numpy()
    target = target.cpu().numpy()
    best_th = 0
    best_score = 0 
    for th in range(0,100):
        i, j = np.where(pred>th/100)
        pred_onehot = np.zeros_like(pred)
        pred_onehot[i,j] = 1
        CorrectPred = np.count_nonzero(np.logical_and(pred_onehot, target), axis=1)
        LabelPred = np.count_nonzero(np.logical_or(pred_onehot, target), axis=1)
        score = np.mean(CorrectPred / LabelPred)
        if (score > best_score):
            best_score = score
            best_th = th
    best_pred_label = [[] for i in range(len(pred))]
    best_i, best_j = np.where(pred>best_th/100)
    for i in range(len(best_i)):
        best_pred_label[best_i[i]].append(best_j[i])
    return best_score, best_th, best_pred_label

def CalcMultiLabelAccuracyWithTh(pred, target, th):
    pred = pred.detach().cpu().numpy()
    target = target.cpu().numpy()
    i, j = np.where(pred>th/100)
    pred_onehot = np.zeros_like(pred)
    pred_onehot[i,j] = 1
    CorrectPred = np.count_nonzero(np.logical_and(pred_onehot, target), axis=1)
    LabelPred = np.count_nonzero(np.logical_or(pred_onehot, target), axis=1)
    score = np.mean(CorrectPred / LabelPred)
    pred_label = [[] for i in range(len(pred))]
    best_i, best_j = np.where(pred>th/100)
    for i in range(len(best_i)):
        pred_label[best_i[i]].append(best_j[i])
    return score, th, pred_label

###GPU or CPU?
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device = " + device.type)

###Label load   
labels = []
with open("train_"+raw_or_feature+"/labels.txt","r") as f:
    reader = f.readlines()
    for i in reader:
        l = i.replace(',\n', '').split(',')
        l[1] = RecpieToIndex(l[1])
        for j in range(2, len(l)):
            l[j] = ActToIndex(l[j])
        labels.append(l)

###Train Data load
print("is_activity_recognition? {}".format(is_activity_recognition))
print("Train Data Load Start...")
for bodypart in bodyparts:
    train_x = []
    train_y = []
    train_len = []
    csvname_train_used = []
#Activity recognition
    for label in labels:
        pattern = re.compile(r'subject[%s]' %users_for_train)
        if bool(re.match(pattern,label[0])):                
            csvname = label[0]
            recipe = label[1]
            activity = label[2:]
            reader = pd.read_csv("train_"+raw_or_feature+"/"+bodypart+"/"+csvname+".csv")
            if(len(reader)>MIN_SEQ_LEN):
                csvname_train_used.append(csvname)
                reader = np.array(reader)
                train_len.append(len(reader))
                train_x.append(torch.tensor(reader[:,:NUM_CHANNEL],dtype=torch.float32))
                if is_activity_recognition:
                    train_y.append(MultiLabelBinarizer(activity, NUM_CLASSES))
                else:
                    train_y.append(recipe)
            else:
                if network_id==1 or network_id==2:
                    csvname_train_used.append(csvname)
                    train_len.append(MIN_SEQ_LEN)
                    train_x.append(torch.tensor([[0 for i in range(NUM_CHANNEL)] for j in range(MIN_SEQ_LEN)],dtype=torch.float32))
                    if is_activity_recognition:
                        train_y.append(MultiLabelBinarizer(activity, NUM_CLASSES))
                    else:
                        train_y.append(recipe)
    if network_id==2 or network_id==4:
        train_x = torch.nn.utils.rnn.pack_sequence(train_x, enforce_sorted=False).to(device)
    else:
        train_x = torch.nn.utils.rnn.pad_sequence(train_x).permute(1,2,0).to(device)
    train_y = torch.tensor(train_y)    
    print(train_y.size())
    if bodypart == bodyparts[0]:
        train_csvname0 = csvname_train_used
        train_x0_gpu = train_x
        train_y0 = train_y
        train_len0_gpu = torch.tensor(train_len).to(device)
        train_csvname0 = csvname_train_used
    elif bodypart == bodyparts[1]:
        train_csvname1 = csvname_train_used
        train_x1_gpu = train_x
        train_y1 = train_y
        train_len1_gpu = torch.tensor(train_len).to(device)
    elif bodypart == bodyparts[2]:
        train_csvname2 = csvname_train_used
        train_x2_gpu = train_x
        train_y2 = train_y
        train_len2_gpu = torch.tensor(train_len).to(device)
    elif bodypart == bodyparts[3]:
        train_csvname3 = csvname_train_used
        train_x3_gpu = train_x
        train_y3 = train_y
        train_len3_gpu = torch.tensor(train_len).to(device)
print("Done")
###Test Data load
print("Test Data Load Start...")
#print("is_activity_recognition? {}".format(is_activity_recognition))
for bodypart in bodyparts:
    test_x = []
    test_y = []
    test_len = []
    csvname_test_used = []
#Activity recognition
    for label in labels:
        pattern = re.compile(r'subject[%s]' %users_for_test)
        if bool(re.match(pattern,label[0])):
            csvname = label[0]
            recipe = label[1]
            activity = label[2:]
#            reader = pd.read_csv("train_"+raw_or_feature+"/"+bodypart+"/"+csvname+".csv")
            reader = pd.read_csv("test_"+raw_or_feature+"/"+bodypart+"/"+csvname+".csv")
            if(len(reader)>MIN_SEQ_LEN):
                csvname_test_used.append(csvname)
                reader = np.array(reader)
                test_len.append(len(reader))
                test_x.append(torch.tensor(reader[:,:NUM_CHANNEL],dtype=torch.float32))
                if is_activity_recognition:
                    test_y.append(MultiLabelBinarizer(activity, NUM_CLASSES))
                else:
                    test_y.append(recipe)
            else:
                if network_id==1 or network_id==2:
                    csvname_test_used.append(csvname)
                    test_len.append(MIN_SEQ_LEN)
                    test_x.append(torch.tensor([[0 for i in range(NUM_CHANNEL)] for j in range(MIN_SEQ_LEN)],dtype=torch.float32))
                    if is_activity_recognition:
                        test_y.append(MultiLabelBinarizer(activity, NUM_CLASSES))
                    else:
                        test_y.append(recipe)
    if network_id==2 or network_id==4:
        test_x = torch.nn.utils.rnn.pack_sequence(test_x, enforce_sorted=False).to(device)
    else:
        test_x = torch.nn.utils.rnn.pad_sequence(test_x).permute(1,2,0).to(device)
    test_y = torch.tensor(test_y)
    print(test_y.size())    
    if bodypart == bodyparts[0]:
        test_csvname0 = csvname_test_used
        test_x0_gpu = test_x
        test_y0 = test_y
        test_len0_gpu = torch.tensor(test_len).to(device)
    elif bodypart == bodyparts[1]:
        test_csvname1 = csvname_test_used
        test_x1_gpu = test_x
        test_y1 = test_y
        test_len1_gpu = torch.tensor(test_len).to(device)
    elif bodypart == bodyparts[2]:
        test_csvname2 = csvname_test_used
        test_x2_gpu = test_x
        test_y2 = test_y
        test_len2_gpu = torch.tensor(test_len).to(device)
    elif bodypart == bodyparts[3]:
        test_csvname3 = csvname_test_used
        test_x3_gpu = test_x
        test_y3 = test_y
        test_len3_gpu = torch.tensor(test_len).to(device)
print("Done")

###Training and testing       
total_loss_train_lists = [[i+1 for i in range(EPOCH_NUM)]]    
accuracy_train_lists = [[i+1 for i in range(EPOCH_NUM)]]
time_train_lists = [[i+1 for i in range(EPOCH_NUM)]]
total_loss_test_lists = [[i+1 for i in range(EPOCH_NUM)]]    
accuracy_test_lists = [[i+1 for i in range(EPOCH_NUM)]]
time_test_lists = [[i+1 for i in range(EPOCH_NUM)]]


for iter in range(NUM_ITER):
    print("Iter{}/{}".format(iter+1,NUM_ITER))
    total_loss_train_list = []
    total_loss_test_list = []
    accuracy_train_list = []
    accuracy_test_list = []
    time_train_list = []
    time_test_list = []
    ###model and loss function
    if network_id==1:
        model = ConvLSTM(seq_size=NUM_CHANNEL, map_size=MAP_SIZE, hidden_size=HIDDEN_SIZE, out_size=NUM_CLASSES)
    elif network_id==2:
        model = LSTM(seq_size=NUM_CHANNEL, hidden_size=HIDDEN_SIZE, out_size=NUM_CLASSES)
    elif network_id==3:
        model = ConvLSTM_single(seq_size=NUM_CHANNEL, map_size=MAP_SIZE, hidden_size=HIDDEN_SIZE, out_size=NUM_CLASSES)
    elif network_id==4:
        model = LSTM_single(seq_size=NUM_CHANNEL, hidden_size=HIDDEN_SIZE, out_size=NUM_CLASSES)
    model.to(device)
    if is_activity_recognition:
        pos_weight = torch.ones([NUM_CLASSES])
        pos_weight_gpu = pos_weight.to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_gpu)
    elif not is_activity_recognition:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
        
    ###Training start
    print("NUM_CLASS:{}\tMAP_SIZE:{}\tHIDDEN_SIZE:{}".format(NUM_CLASSES,MAP_SIZE,HIDDEN_SIZE))
    print("Training starts")
    if iter == 0:
        csvname_train_used = train_csvname0
        csvname_test_used = test_csvname0
        train_y = train_y0
        test_y = test_y0
    elif iter == 1:
        csvname_train_used = train_csvname1
        csvname_test_used = test_csvname1        
        train_y = train_y1
        test_y = test_y1
    elif iter == 2:
        csvname_train_used = train_csvname2
        csvname_test_used = test_csvname2
        train_y = train_y2
        test_y = test_y2
    elif iter == 3:
        csvname_train_used = train_csvname3
        csvname_test_used = test_csvname3
        train_y = train_y3        
        test_y = test_y3
    csvname_test_used.append("Total Train loss")
    csvname_test_used.append("Train Accuracy")
    csvname_test_used.append("Training time")
    csvname_test_used.append("Total Test loss")
    csvname_test_used.append("Test Accuracy")
    csvname_test_used.append("Testing time")
    csvname_test_used.append("threshold sigmoid train")
    csvname_test_used.append("threshold sigmoid test")
    st = datetime.datetime.now()
    FinalPred = []
    FinalSigmoid = []
    FinalPred.append(csvname_test_used)
    FinalSigmoid.append(csvname_test_used)
    if is_activity_recognition:
        train_y_debinarized = MultiLabelDebinarizer(train_y)
        test_y_debinarized = MultiLabelDebinarizer(test_y)
    elif not is_activity_recognition:
        train_y_debinarized = train_y.numpy().tolist()
        test_y_debinarized = test_y.numpy().tolist()
    for i in range(8):
        test_y_debinarized.append("-----")
    FinalPred.append(test_y_debinarized)
    FinalSigmoid.append(test_y_debinarized)
    header = []
    header.append("Filename")
    header.append("Groundtruth")
    total_time_train = datetime.timedelta(0)
    total_time_test = datetime.timedelta(0)
    for epoch in range(EPOCH_NUM):
        total_loss_train = 0
        #Train mode
        model.train()
        st = datetime.datetime.now()
        optimizer.zero_grad()
        if network_id==1:
            y = model(train_x0_gpu,train_x1_gpu,train_x2_gpu,train_x3_gpu,train_len0_gpu,train_len1_gpu,train_len2_gpu,train_len3_gpu)
        elif network_id==2:
            y = model(train_x0_gpu,train_x1_gpu,train_x2_gpu,train_x3_gpu)
        elif network_id==3:
            if iter == 0:
                y = model(train_x0_gpu,train_len0_gpu)
            elif iter == 1:
                y = model(train_x1_gpu,train_len1_gpu)
            elif iter == 2:
                y = model(train_x2_gpu,train_len2_gpu)
            elif iter == 3:
                y = model(train_x3_gpu,train_len3_gpu)
        elif network_id==4:
            if iter == 0:
                y = model(train_x0_gpu)
            elif iter == 1:
                y = model(train_x1_gpu)
            elif iter == 2:
                y = model(train_x2_gpu)
            elif iter == 3:                
                y = model(train_x3_gpu)
        train_y_gpu = train_y.to(device)
        loss_train = criterion(y, train_y_gpu)
#        loss_train.backward()
        total_loss_train += loss_train.item()
#        optimizer.step()           
        y = torch.sigmoid(y)
        if is_activity_recognition:
            accuracy_train, th, pred = CalcMultiLabelAccuracy(y, train_y)
        elif not is_activity_recognition:
            _, y_ = torch.max(y, 1)
            pred = y_.cpu().numpy().tolist()
            accuracy_train = np.count_nonzero(y_.cpu()==train_y)/len(y_)
            th = "---"
        ed = datetime.datetime.now()
        total_loss_train_list.append(total_loss_train)
        accuracy_train_list.append(accuracy_train)
        time_train = ed - st
        total_time_train = total_time_train + time_train
        time_train_list.append(time_train)
        #Test mode
        model.eval()
        st = datetime.datetime.now()      
        if network_id==1:
            y = model(test_x0_gpu, test_x1_gpu, test_x2_gpu, test_x3_gpu,test_len0_gpu, test_len1_gpu, test_len2_gpu, test_len3_gpu)
            test_y= test_y0
        elif network_id==2:
            y = model(test_x0_gpu, test_x1_gpu, test_x2_gpu, test_x3_gpu)
            test_y= test_y0
        elif network_id==3:
            if iter == 0:
                y = model(test_x0_gpu,test_len0_gpu)
                test_y = test_y0
            elif iter == 1:
                y = model(test_x1_gpu,test_len1_gpu)
                test_y = test_y1 
            elif iter == 2:
                y = model(test_x2_gpu,test_len2_gpu)
                test_y = test_y2
            elif iter ==3:
                y = model(test_x3_gpu,test_len3_gpu)
                test_y = test_y3
        elif network_id==4:
            if iter == 0:                
                y = model(test_x0_gpu)
                test_y = test_y0
            elif iter == 1:
                y = model(test_x1_gpu)
                test_y = test_y1
            elif iter == 2:
                y = model(test_x2_gpu)
                test_y = test_y2
            elif iter == 3:
                y = model(test_x3_gpu)
                test_y = test_y3
        test_y_gpu = test_y.to(device)
        loss_test = criterion(y, test_y_gpu)
        total_loss_test = loss_test.item()
        model.train()
        loss_train.backward()
        optimizer.step()  
        y = torch.sigmoid(y)
        if is_activity_recognition:
            accuracy_test, th_test, pred = CalcMultiLabelAccuracyWithTh(y, test_y, th)
        elif not is_activity_recognition:
            _, y_ = torch.max(y, 1)
            pred = y_.cpu().numpy().tolist()
            accuracy_test = np.count_nonzero(y_.cpu()==test_y)/len(y_)
            th_test = "---"
        ed = datetime.datetime.now()
        total_loss_test_list.append(total_loss_test)
        accuracy_test_list.append(accuracy_test)
        time_test = ed - st
        total_time_test = total_time_test + time_test
        time_test_list.append(time_test)
        if (epoch+1) % 100 == 0:
            header.append("EPOCH"+str(epoch+1))
            pred.append(total_loss_train)
            pred.append(accuracy_train)
            pred.append(total_time_train)
            pred.append(total_loss_test)
            pred.append(accuracy_test)
            pred.append(total_time_test)
            pred.append(th)
            pred.append(th_test)
            FinalPred.append(pred)
            sigmoid = y.detach().cpu().numpy().tolist()
            sigmoid.append(total_loss_train)
            sigmoid.append(accuracy_train)
            sigmoid.append(total_time_train)
            sigmoid.append(total_loss_test)
            sigmoid.append(accuracy_test)
            sigmoid.append(total_time_test)
            sigmoid.append(th)
            sigmoid.append(th_test)
            FinalSigmoid.append(sigmoid)
            if is_activity_recognition:
                print("epoch:{}\ttrain loss:{:.4f}\ttrain accu:{:.4f}\ttrain time:{}\ttest loss:{:.4f}\ttest accu:{:.4f}\ttest time:{}\tth train:{}\tth test:{}".format(epoch+1,total_loss_train,accuracy_train,total_time_train,total_loss_test,accuracy_test,total_time_test,th,th_test))
            elif not is_activity_recognition:                         
                print("epoch:{}\ttrain loss:{:.4f}\ttrain accu:{:.4f}\ttrain time:{}\ttest loss:{:.4f}\ttest accu:{:.4f}\ttest time:{}".format(epoch+1,total_loss_train,accuracy_train,total_time_train,total_loss_test,accuracy_test,total_time_test))
            total_time_train = datetime.timedelta(0)
            total_time_test = datetime.timedelta(0)
    total_loss_train_lists.append(total_loss_train_list)
    accuracy_train_lists.append(accuracy_train_list)
    time_train_lists.append(time_train_list)
    total_loss_test_lists.append(total_loss_test_list)
    accuracy_test_lists.append(accuracy_test_list)
    time_test_lists.append(time_test_list)
    path = "./result_"+("Feature" if raw_or_feature == "feature" else "Rawdata")
    path = path + ("/ConvLSTM" if network_id == 1 else "/LSTM" if network_id == 2 else "/ConvLSTMsingle" if network_id == 3 else "/LSTMsingle")
    path = path + ("/Activity" if is_activity_recognition else "/Recipe")
    path = path + "/Train"+users_for_train+"Test"+users_for_test
    path = path + "/HIDDEN"+str(HIDDEN_SIZE)+"MAP"+str(MAP_SIZE)
    os.makedirs(path, exist_ok=True)
    f = open(path + "/prediction"+bodyparts[iter]+".tsv", "w")
    for i in header:
        f.write(i+"\t")
    f.write("\n")
    for i in range(len(FinalPred[0])):
        for j in range(len(FinalPred)):
            f.write(str(FinalPred[j][i])+"\t")
        f.write("\n")
    f.close()
    g = open(path + "/sigmoid"+bodyparts[iter]+".tsv", "w")
    for i in header:
        g.write(i+"\t")
    g.write("\n")
    for i in range(len(FinalSigmoid[0])):
        for j in range(len(FinalSigmoid)):
            g.write(str(FinalSigmoid[j][i])+"\t")
        g.write("\n")
    g.close()
h = open(path + "/total_loss_train.pickle","wb")
pickle.dump(total_loss_train_lists,h)
h.close
h = open(path + "/accuracy_train.pickle","wb")
pickle.dump(accuracy_train_lists,h)
h.close
h = open(path + "/time_train.pickle","wb")
pickle.dump(time_train_lists,h)
h.close
h = open(path +  "/total_loss_test.pickle","wb")
pickle.dump(total_loss_test_lists,h)
h.close
h = open(path + "/accuracy_test.pickle","wb")
pickle.dump(accuracy_test_lists,h)
h.close
h = open(path + "/time_test.pickle","wb")
pickle.dump(time_test_lists,h)
h.close