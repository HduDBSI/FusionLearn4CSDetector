import torch
import torch.nn as nn
import torch.optim as optim
from gensim.models.word2vec import Word2Vec
import numpy as np
import pandas as pd
import time
from focalloss import FocalLoss
from preprocess_data import preprocess_data

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve
from model import ASTNN



EPOCH = 30

projectList = ['Ant','flink','gradle','hadoop','hbase','jruby','kafka','mockito','storm','tomcat']
projectList = ['Ant','jruby','kafka','mockito','storm','tomcat']

def get_batch(dataset, i, batch_size):
    return dataset.iloc[i: i + batch_size]

#projectname = projectList[9]

def init_data_and_model(projectname):
    print("projectname",projectname)

    preprocess_data(projectname)

    train_labels = "/home/yqx/Downloads/DesigniteJava-master/myData/"+projectname+"/methods/trainlabels.txt"
    test_labels = "/home/yqx/Downloads/DesigniteJava-master/myData/"+projectname+"/methods/testlabels.txt"

    print('Reading data...')
    w2v = Word2Vec.load('./data/w2v_128').wv
    #embeddings = torch.tensor(np.vstack([w2v.vectors, [0] * 128]))

    programs = pd.read_pickle('./data/programs.pkl')

    #print("programs",programs)
    train_labels = open(train_labels).readlines()
    test_labels = open(test_labels).readlines()
    train_code_name = []
    test_code_name = []
    for line in train_labels:
        train_code_name.append(line.split('    ')[0])
    for line in test_labels:
        test_code_name.append(line.split('    ')[0])

    train_set_indexTree = []
    train_set_label = []
    test_set_indexTree = []
    test_set_label = []
    for i,name in enumerate(programs['name']):
        if programs['name'][i] in train_code_name and programs['index_tree'][i] != []:
            train_set_indexTree.append(programs['index_tree'][i])
            train_set_label.append(programs['label'][i])
        elif programs['name'][i] in test_code_name and programs['index_tree'][i] != []:
            test_set_indexTree.append(programs['index_tree'][i])
            test_set_label.append(programs['label'][i])

    training_data = {"index_tree":train_set_indexTree,"label":train_set_label}
    training_set = pd.DataFrame(training_data, columns = ['index_tree','label'])

    test_data = {"index_tree":test_set_indexTree,"label":test_set_label}
    test_set = pd.DataFrame(test_data, columns = ['index_tree','label'])

    training_set.to_pickle('./data/train_programs.pkl')
    test_set.to_pickle('./data/test_programs.pkl')
    training_set=pd.read_pickle('./data/train_programs.pkl')
    test_set=pd.read_pickle('./data/test_programs.pkl')
    validation_set = test_set
    
    net = ASTNN(output_dim=2,
            embedding_dim=128, num_embeddings=len(w2v.vectors) + 1, embeddings=None,
            batch_size=BATCH_SIZE).cuda()
    

    #criterion = nn.CrossEntropyLoss()
    criterion = FocalLoss().cuda()
    optimizer = torch.optim.Adamax(net.parameters())
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=5e-4)

    return net,criterion,optimizer,training_set,validation_set,test_set


def train(dataset, backward=True):
    total_acc = 0.0
    total_loss = 0.0
    total = 0.0
    i = 0
    y_preds = []
    y_trues = []
    
    batchloss = 0
    while i < len(dataset):
        data = get_batch(dataset, i, BATCH_SIZE)
        input, label = data['index_tree'], torch.Tensor([[0,1] if label==1 else [1,0] for label in data['label']]).cuda()
        tlabel = [label for label in data['label']]
        i += BATCH_SIZE
        
        #print(" label",label.shape)
        net.zero_grad()
        net.batch_size = len(input)
        
        output = net(input)

        _, predicted = torch.max(output.data, 1)
        

        loss = criterion(output, label)
        batchloss += loss
        if backward and i%64==0:
            #print(i)
            batchloss.backward()
            #loss.backward(retain_graph = True)
            optimizer.step()
            batchloss = 0

        # calc acc
        #pred = output.data.argmax(1)
        #correct = pred.eq(label).sum().item()
        correct = torch.sum(torch.eq(torch.argmax(output, dim=1), torch.argmax(label, dim=1)))
        total_acc += correct
        total += len(input)
        total_loss += loss.item() * len(input)
        #print("tlabel",tlabel)
        #print("predicted.tolist()",predicted.tolist())
        #exit()
        y_trues += tlabel
        y_preds += predicted.tolist()
        r=recall_score(y_trues, y_preds)
        p=precision_score(y_trues, y_preds)
        f1=f1_score(y_trues, y_preds)
        #acc = accuracy_score(y_trues, y_preds)
        #matrix = confusion_matrix(y_trues, y_preds)
        

    return total_loss / total, total_acc / total, p, r, f1
import random
def Undersampling(training_set):
    print('training_set',len(training_set['label']))
    pos = 0
    neg = 0
    rate = 0.5
    posSamples = []
    negSamples = []
    selectSamples = []
    for i,label in enumerate(training_set['label']):
        #print('label',label)
        if label == 1:
            pos+=1
            posSamples.append([training_set['index_tree'][i],label])
        else:
            neg+=1
            negSamples.append([training_set['index_tree'][i],label])
    print('sample ratio(pos:neg): ',pos,':',neg)
    if pos>=neg:
        sampleNum = int(neg*rate)
        selectSamples = negSamples[:sampleNum] + posSamples[:sampleNum]
        #selectSamples = negSamples + posSamples[:neg]
        print('after sampling(pos:neg): ',sampleNum,':',sampleNum)
    elif neg>pos:
        sampleNum = int(pos*rate)
        selectSamples = negSamples[:sampleNum] + posSamples[:sampleNum]
        #selectSamples = negSamples[:pos] + posSamples
        print('after sampling(pos:neg): ',sampleNum,':',sampleNum)
    random.shuffle(selectSamples)
    index_tree = []
    label = []
    for item in selectSamples:
        index_tree.append(item[0])
        label.append(item[1])
    data = {'index_tree':index_tree, 'label':label}
    Sampled_set = pd.DataFrame(data,columns = ['index_tree','label'])
    return Sampled_set

    
if __name__ == '__main__':
    
    
    #三次求平均值
    for projectname in projectList:
        BATCH_SIZE = 1
        p_sum = r_sum = f1_sum = 0
        #test_result = "/home/yqx/Desktop/TD-data-process/outData/"+projectname+"/"+projectname+"_ASTNN_1_.txt"
        test_result = "/home/yqx/Downloads/DesigniteJava-master/myData/"+projectname+"/methods_result_.txt"
        num = 1.0
        for i in range(int(num)):
            net,criterion,optimizer,training_set,validation_set,test_set = init_data_and_model(projectname)
            print('Start Training...')
            
            for epoch in range(EPOCH):
                start_time = time.time()
                #print(len(training_set),training_set)
                training_data = Undersampling(training_set)

                training_loss, training_acc, p, r, f1 = train(training_data)
                #validation_loss, validation_acc, p, r, f1 = train(validation_set, backward=False)

                end_time = time.time()
                print('[Epoch: %2d/%2d] Train Loss: %.4f, Train Acc: %.3f, Time Cost: %.3f s, \
                p=%.4g,r=%.4g,f1=%.4g'
                    % (epoch + 1, EPOCH, training_loss, training_acc, end_time - start_time, p, r, f1))

                #torch.save(net.state_dict(), './data/params_epoch[%d].pkl' % (epoch + 1))

            test_loss, test_acc, p, r, f1 = train(test_set, backward=False)
            print('Test Acc: %.3f, p=%.4g, r=%.4g, f1=%.4g' % (test_acc, p, r, f1))
            test_p_r_f1 = open(test_result, 'a')
            test_p_r_f1.write(projectname+str(epoch) +" "+ str(p) +" "+ str(r) +" "+ str(f1)+"\n")
            test_p_r_f1.close()
            p_sum += p
            r_sum += r
            f1_sum += f1

            torch.save(net.state_dict(), './data/'+ projectname +'.pkl')
            print('Saved model parameters at', './data/'+ projectname +'.pkl')
        test_p_r_f1 = open(test_result, 'a')
        test_p_r_f1.write("ASTNN-- "+projectname+" 平均值： "+ str(p_sum/num) +" "+ str(r_sum/num) +" "+ str(f1_sum/num)+"\n")
        test_p_r_f1.close()

    '''
    #Test
    BATCH_SIZE = 1
    projectname = 'tomcat'
    test_result = "/home/yqx/Downloads/DesigniteJava-master/myData/"+projectname+"/type_result_.txt"
    net,criterion,optimizer,training_set,validation_set,test_set = init_data_and_model(projectname)

    #b=torch.load('./data/params_epoch[30].pkl')
    #torch.save(b,'./data/params_epoch[30].pkl')
    net.load_state_dict(torch.load('./data/'+ projectname +'.pkl'),False)

    #net = torch.load('./data/params_epoch[30].pkl')
    #torch.load(net.state_dict(), './data/params_epoch[30].pkl')
    test_loss, test_acc, p, r, f1 = train(test_set, backward=False)
    print('Test Acc: %.3f, p=%.4g, r=%.4g, f1=%.4g' % (test_acc, p, r, f1))
    test_p_r_f1 = open(test_result, 'a')
    test_p_r_f1.write(projectname+str(30) +" "+ str(p) +" "+ str(r) +" "+ str(f1)+"\n")
    test_p_r_f1.close()
    '''