import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm, trange
import argparse
import random
import os
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve
from mymodel import MyModel
from data_loader import getTwoDicts, getAllData2Dict
from focalloss import FocalLoss

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--lr_decay_step_size', type=int, default=50)
parser.add_argument('--hidden', type=int, default=128)
parser.add_argument('--d_model', type=int, default=128)
parser.add_argument('--num_encoder_layers', type=int, default=3)
parser.add_argument('--dim_feedforward', type=int, default=512)
parser.add_argument('--dropout', type=int, default=0.1)
parser.add_argument('--hidden_dropout_prob', type=int, default=0.1)
parser.add_argument('--attention_probs_dropout_prob', type=int, default=0.1)
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--nhead', type=int, default=8)
parser.add_argument('--num_attention_heads', type=int, default=8)
parser.add_argument('--alpha', type=int, default=0.2)
parser.add_argument("--threshold", default=0)
args = parser.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_10_fold_cross_train_test_list(project):
    trainlist = []
    testlist = []
    for projectname in projectList:
        if projectname != project:
            train_list_path = "/home/yqx/Downloads/DesigniteJava-master/myData/"+projectname+"/types/trainlabels.txt"
            trainlist += open(train_list_path).readlines()
            train_list_path = "/home/yqx/Downloads/DesigniteJava-master/myData/"+projectname+"/types/testlabels.txt"
            trainlist += open(train_list_path).readlines()
        else:
            test_list_path = "/home/yqx/Downloads/DesigniteJava-master/myData/"+projectname+"/types/trainlabels.txt"
            testlist += open(test_list_path).readlines()
            test_list_path = "/home/yqx/Downloads/DesigniteJava-master/myData/"+projectname+"/types/testlabels.txt"
            testlist += open(test_list_path).readlines()
    
    new_trainlist = []
    new_testlist = []
    for line in trainlist:
        #print(line.split('    '))

        if line.split('    ')[1] == '0\n':
            new_trainlist.append(line.split('    ')[0]+'    '+'0')
        else:
            new_trainlist.append(line.split('    ')[0]+'    '+'1')
    for line in testlist:
        #print(line.split('    '))
        if line.split('    ')[1] == '0\n':
            new_testlist.append(line.split('    ')[0]+'    '+'0')
        else:
            new_testlist.append(line.split('    ')[0]+'    '+'1')
    #print(new_trainlist)
    return new_trainlist, new_testlist
    
def init_data_and_model(project, projectList):
    jsonFolderPathList = []
    test_result = '/home/yqx/Downloads/DesigniteJava-master/myData/'+ project +'/_types_10_cross_.txt'
    for projectname in projectList:
        jsonFolderPathList.append('/home/yqx/Downloads/DesigniteJava-master/myData/'+ projectname +'/types/rawjson')
    trainlist, testlist = get_10_fold_cross_train_test_list(project)
    print('trainlist:testlist',len(trainlist),len(testlist))
    vocabdict, metricdict = getTwoDicts(jsonFolderPathList)
    
    vocablen, metrilen = len(vocabdict), len(metricdict)
    print('vocablen, metrilen',vocablen, metrilen)
    allJsonDict = getAllData2Dict(jsonFolderPathList, vocabdict, metricdict)
    print('allJsonDict',len(allJsonDict))
    model = MyModel(vocablen, metrilen, args.hidden, args.d_model, args.nhead, args.num_encoder_layers,
            args.dim_feedforward, args.dropout, args.alpha).to(device)
    return test_result, trainlist, testlist, allJsonDict, model

#test_result, trainlist, testlist, allJsonDict, model = init_data_and_model(project)


criterion = FocalLoss().to(device)

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def getBatchList(allJsonDict,line_list):
    batchlist = []
    for line in line_list:
        try:
            jsonName = line.split('    ')[0][:-5]+'.json'
            label = int(line.split('    ')[1])
            data = allJsonDict[jsonName]
            x = data['x']
            edge_index = data['edge_index']
            edge_attr = data['edge_attr']
            metrics = data['metrics']
            token_list = data['token_list']
            src_metrics = data['src_metrics']
            
            a_sample = [x, edge_index, edge_attr, metrics, token_list, src_metrics, label]
            
            batchlist.append(a_sample)
        except:
            #print(jsonName)
            continue
    #print('batchlist',len(batchlist))
    return batchlist


def getBatch(line_list, batch_size, batch_index, device):
    start_line = batch_size*batch_index
    end_line = start_line+batch_size
    dataList = getBatchList(allJsonDict,line_list[start_line:end_line])
    return dataList

def split_batch(init_list, batch_size):
    groups = zip(*(iter(init_list),) * batch_size)
    end_list = [list(i) for i in groups]
    count = len(init_list) % batch_size
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list

def test(testlist, model_index, allJsonDict, batch_size):
    with torch.no_grad():
        
        #model.load_state_dict(torch.load('./model/epoch'+str(model_index)+'.pkl'))
        model.eval()

        notFound = 0
        testCount = 0
        y_preds = []
        y_trues = []
        batches = split_batch(testlist, batch_size)
        Test_data_batches = trange(len(batches), leave=True, desc = "Test")
        for i in Test_data_batches:
            #lable
            line_info = batches[i][0].split('    ')
            jsonName = line_info[0][:-5]+'.json'
            label = int(line_info[1])
            #data
            try:
                data = allJsonDict[jsonName]
                x = data['x']
                edge_index = data['edge_index']
                edge_attr = data['edge_attr']
                metrics = data['metrics']
                token_list = data['token_list']
                src_metrics = data['src_metrics']
                testCount += 1
            except:
                notFound += 1
            #predict
            output = model(x, edge_index, edge_attr, metrics, token_list, src_metrics)
            _, predicted = torch.max(output.data, 1)
            
            y_trues += [label]
            y_preds += predicted.tolist()

            r=recall_score(y_trues, y_preds)
            p=precision_score(y_trues, y_preds)
            f1=f1_score(y_trues, y_preds)
            acc = accuracy_score(y_trues, y_preds)
            matrix = confusion_matrix(y_trues, y_preds)

            Test_data_batches.set_description("Test (p=%.4g,r=%.4g,f1=%.4g)" % (p, r, f1))
        #print("testCount",testCount)
        #print("notFound",notFound)
        #print("acc",acc)
        #print("matrix",type(matrix),matrix)
        #print("tp,tn,fp,fn:",matrix[1][1],matrix[0][0],matrix[0][1],matrix[1][0])
        p = float(format(p, '.4f'))
        r = float(format(r, '.4f'))
        f1 = float(format(f1, '.4f'))
        print("\n p, r, f1:", p, r, f1)
        return p, r, f1

def Undersampling(trainlist):
    print('trainlist',len(trainlist))
    random.shuffle(trainlist)
    pos = 0
    neg = 0
    rate = 0.2
    posSamples = []
    negSamples = []
    selectSamples = []
    for sample in trainlist:
        if sample.split('    ')[1] == '0':
            neg+=1
            negSamples.append(sample)
        else:
            pos+=1
            posSamples.append(sample)
    print('sample ratio(pos:neg): ',pos,':',neg)
    if pos>=neg:
        sampleNum = int(neg*rate)
        selectSamples = negSamples[:sampleNum] + posSamples[:sampleNum]
    elif neg>pos:
        sampleNum = int(pos*rate)
        selectSamples = negSamples[:sampleNum] + posSamples[:sampleNum]
    random.shuffle(selectSamples)
    pos = 0
    neg = 0
    for item in selectSamples:
        if item.split('    ')[1] == '0':
            neg+=1
        else:
            pos+=1
    print('after sampling(pos:neg): ',pos,':',neg)
    return selectSamples

def train(trainlist):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()
    #print("loaded ", './saveModel/epoch'+str(start_train_model_index)+'.pkl')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    print("???????????????", get_parameter_number(model))
    print("nhead ", args.nhead," batch_size ", args.batch_size)
    print("dropout = ",args.dropout)

    #trainlist = trainlist[:int(len(trainlist)*0.2)]
    epochs = trange(args.epochs, leave=True, desc = "Epoch")
    iterations = 0
    for epoch in epochs:
        #print(epoch)
        totalloss=0.0
        main_index=0.0
        #batches = create_batches(trainDataList)
        #for index, batch in tqdm(enumerate(batches), total=len(batches), desc = "Batches"):
        count = 0
        right = 0
        acc = 0
        
        train_data = Undersampling(trainlist)
        #random.shuffle(trainlist)

        for batch_index in tqdm(range(int(len(train_data)/args.batch_size))):
            batch = getBatch(train_data, args.batch_size, batch_index, device)
            optimizer.zero_grad()
            
            batchloss= 0
            for data in batch:
                model.train()
                x, edge_index, edge_attr, metrics, token_list, src_metrics, label = data
                #print("data",data)
                #print(type(label),label)
                label=torch.Tensor([[0,1]]).to(device) if label==1 else torch.Tensor([[1,0]]).to(device)
                #print("label ",label.device," ",label)
                output = model(x, edge_index, edge_attr, metrics, token_list, src_metrics)
                #print("output",output)
                #print("label",label)

                batchloss = batchloss + criterion(output, label)

                count += 1
                right += torch.sum(torch.eq(torch.argmax(output, dim=1), torch.argmax(label, dim=1)))
            #print("batchloss",batchloss)
            acc = right*1.0/count
            batchloss.backward(retain_graph = True)
            optimizer.step()
            loss = batchloss.item()
            totalloss += loss
            main_index = main_index + len(batch)
            loss = totalloss/main_index
            epochs.set_description("Epoch (Loss=%g) (Acc = %g)" % (round(loss,5) , acc))
            iterations += 1
        '''
        if (epoch)%10==0:
            torch.save(model.state_dict(), saveModelsPath+'/'+modelName+'epoch'+str(epoch)+'.pkl')
        '''  
    p, r, f1 = test(testlist, epoch, allJsonDict, 1)
    test_p_r_f1 = open(test_result, 'a')
    test_p_r_f1.write(modelName+str(epoch) +" "+ str(p) +" "+ str(r) +" "+ str(f1)+"\n")
    test_p_r_f1.close()
    return p,r,f1

if __name__ == '__main__':
    #projectList = ['Ant','flink','gradle','hadoop','hbase','jruby','kafka','mockito','storm','tomcat']
    projectList = ['Ant','jruby','kafka','mockito','storm','tomcat']
    model_index = 1
    modelNames = [' gcn_trainsformer_fusion ',' DNN ',' gcn_trainsformer_fusion_v2 ',' Dense(CNN) ',' BiLSTM+attention ']
    modelName = modelNames[model_index]
    num = 3.0
    for project in projectList:
        p_sum = r_sum = f1_sum = 0
        for i in range(int(num)):
            print("projectName:",project)
            test_result, trainlist, testlist, allJsonDict, model = init_data_and_model(project,projectList)
            p,r,f1 = train(trainlist)
            p_sum += p
            r_sum += r
            f1_sum += f1
        test_p_r_f1 = open(test_result, 'a')
        test_p_r_f1.write(modelName +" "+ str(p_sum/num) +" "+ str(r_sum/num) +" "+ str(f1_sum/num)+"\n")
        test_p_r_f1.close()