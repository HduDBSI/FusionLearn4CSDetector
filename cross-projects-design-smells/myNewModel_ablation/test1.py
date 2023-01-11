import os

projectList = ['ant-rel-1.10.12','dubbo-dubbo-3.0.6','hadoop-release-3.3.2-RC5','jfreechart-1.5.3',
    'jmeter-rel-v5.4.3','jruby-9.3.3.0','kafka-3.1.0','mockito-4.4.0','neo4j-4.4.4','tomcat-10.0.18']
    
jsonFolderPath = '/home/yqx/Desktop/TD-data-process/outData'

for root ,dirs, files in os.walk(jsonFolderPath):
    for file in files:
        if not file.endswith('.json'):
            print(file)


project = projectList[2]

def get_10_fold_cross_train_test_list(project):
    trainlist = []
    testlist = []
    for projectname in projectList:
        if projectname != project:
            train_list_path = "/home/yqx/Desktop/TD-data-process/outData/"+projectname+"/typelabels.txt"
            trainlist += open(train_list_path).readlines()
        else:
            test_list_path = "/home/yqx/Desktop/TD-data-process/outData/"+projectname+"/typelabels.txt"
            testlist += open(test_list_path).readlines()
    
    new_trainlist = []
    new_testlist = []
    for line in trainlist:
        
        if line.split()[1] == '0':
            new_trainlist.append(line.split()[0]+'    '+'0')
        else:
            new_trainlist.append(line.split()[0]+'    '+'1')
    for line in testlist:
        if line.split()[1] == '0':
            new_testlist.append(line.split()[0]+'    '+'0')
        else:
            new_testlist.append(line.split()[0]+'    '+'1')

    return new_trainlist, new_testlist



trainlist, testlist = get_10_fold_cross_train_test_list(project)
print('trainlist',len(trainlist))
print('testlist',len(testlist))