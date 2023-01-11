# -*- coding: utf-8 -*-
"""
@Env: Python2.7
@Time: 2019/10/24 13:31
@Author: zhaoxingfeng
@Function：Random Forest（RF），随机森林二分类
@Version: V1.2
参考文献：
[1] UCI. wine[DB/OL].https://archive.ics.uci.edu/ml/machine-learning-databases/wine.
"""
from nbformat import write
import pandas as pd
import numpy as np
import random
import math
import collections
from sklearn.externals.joblib import Parallel, delayed


class Tree(object):
    """定义一棵决策树"""
    def __init__(self):
        self.split_feature = None
        self.split_value = None
        self.leaf_value = None
        self.tree_left = None
        self.tree_right = None

    def calc_predict_value(self, dataset):
        """通过递归决策树找到样本所属叶子节点"""
        if self.leaf_value is not None:
            return self.leaf_value
        elif dataset[self.split_feature] <= self.split_value:
            return self.tree_left.calc_predict_value(dataset)
        else:
            return self.tree_right.calc_predict_value(dataset)

    def describe_tree(self):
        """以json形式打印决策树，方便查看树结构"""
        if not self.tree_left and not self.tree_right:
            leaf_info = "{leaf_value:" + str(self.leaf_value) + "}"
            return leaf_info
        left_info = self.tree_left.describe_tree()
        right_info = self.tree_right.describe_tree()
        tree_structure = "{split_feature:" + str(self.split_feature) + \
                         ",split_value:" + str(self.split_value) + \
                         ",left_tree:" + left_info + \
                         ",right_tree:" + right_info + "}"
        return tree_structure


class RandomForestClassifier(object):
    def __init__(self, n_estimators=10, max_depth=-1, min_samples_split=2, min_samples_leaf=1,
                 min_split_gain=0.0, colsample_bytree=None, subsample=0.8, random_state=None):
        """
        随机森林参数
        ----------
        n_estimators:      树数量
        max_depth:         树深度，-1表示不限制深度
        min_samples_split: 节点分裂所需的最小样本数量，小于该值节点终止分裂
        min_samples_leaf:  叶子节点最少样本数量，小于该值叶子被合并
        min_split_gain:    分裂所需的最小增益，小于该值节点终止分裂
        colsample_bytree:  列采样设置，可取[sqrt、log2]。sqrt表示随机选择sqrt(n_features)个特征，
                           log2表示随机选择log(n_features)个特征，设置为其他则不进行列采样
        subsample:         行采样比例
        random_state:      随机种子，设置之后每次生成的n_estimators个样本集不会变，确保实验可重复
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth if max_depth != -1 else float('inf')
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_split_gain = min_split_gain
        self.colsample_bytree = colsample_bytree
        self.subsample = subsample
        self.random_state = random_state
        self.trees = None
        self.feature_importances_ = dict()

    def fit(self, dataset, targets):
        """模型训练入口"""
        assert targets.unique().__len__() == 2, "There must be two class for targets!"
        targets = targets.to_frame(name='label')

        if self.random_state:
            random.seed(self.random_state)
        random_state_stages = random.sample(range(self.n_estimators), self.n_estimators)

        # 两种列采样方式
        if self.colsample_bytree == "sqrt":
            self.colsample_bytree = int(len(dataset.columns) ** 0.5)
        elif self.colsample_bytree == "log2":
            self.colsample_bytree = int(math.log(len(dataset.columns)))
        else:
            self.colsample_bytree = len(dataset.columns)

        # 并行建立多棵决策树
        self.trees = Parallel(n_jobs=-1, verbose=0, backend="threading")(
            delayed(self._parallel_build_trees)(dataset, targets, random_state)
                for random_state in random_state_stages)
        
    def _parallel_build_trees(self, dataset, targets, random_state):
        """bootstrap有放回抽样生成训练样本集，建立决策树"""
        subcol_index = random.sample(dataset.columns.tolist(), self.colsample_bytree)
        dataset_stage = dataset.sample(n=int(self.subsample * len(dataset)), replace=True, 
                                        random_state=random_state).reset_index(drop=True)
        dataset_stage = dataset_stage.loc[:, subcol_index]
        targets_stage = targets.sample(n=int(self.subsample * len(dataset)), replace=True, 
                                        random_state=random_state).reset_index(drop=True)

        tree = self._build_single_tree(dataset_stage, targets_stage, depth=0)
        #print(tree.describe_tree())
        return tree

    def _build_single_tree(self, dataset, targets, depth):
        """递归建立决策树"""
        # 如果该节点的类别全都一样/样本小于分裂所需最小样本数量，则选取出现次数最多的类别。终止分裂
        if len(targets['label'].unique()) <= 1 or dataset.__len__() <= self.min_samples_split:
            tree = Tree()
            tree.leaf_value = self.calc_leaf_value(targets['label'])
            return tree

        if depth < self.max_depth:
            best_split_feature, best_split_value, best_split_gain = self.choose_best_feature(dataset, targets)
            left_dataset, right_dataset, left_targets, right_targets = \
                self.split_dataset(dataset, targets, best_split_feature, best_split_value)

            tree = Tree()
            # 如果父节点分裂后，左叶子节点/右叶子节点样本小于设置的叶子节点最小样本数量，则该父节点终止分裂
            if left_dataset.__len__() <= self.min_samples_leaf or \
                    right_dataset.__len__() <= self.min_samples_leaf or \
                    best_split_gain <= self.min_split_gain:
                tree.leaf_value = self.calc_leaf_value(targets['label'])
                return tree
            else:
                # 如果分裂的时候用到该特征，则该特征的importance加1
                self.feature_importances_[best_split_feature] = \
                    self.feature_importances_.get(best_split_feature, 0) + 1

                tree.split_feature = best_split_feature
                tree.split_value = best_split_value
                tree.tree_left = self._build_single_tree(left_dataset, left_targets, depth+1)
                tree.tree_right = self._build_single_tree(right_dataset, right_targets, depth+1)
                return tree
        # 如果树的深度超过预设值，则终止分裂
        else:
            tree = Tree()
            tree.leaf_value = self.calc_leaf_value(targets['label'])
            return tree

    def choose_best_feature(self, dataset, targets):
        """寻找最好的数据集划分方式，找到最优分裂特征、分裂阈值、分裂增益"""
        best_split_gain = 1
        best_split_feature = None
        best_split_value = None

        for feature in dataset.columns:
            if dataset[feature].unique().__len__() <= 100:
                unique_values = sorted(dataset[feature].unique().tolist())
            # 如果该维度特征取值太多，则选择100个百分位值作为待选分裂阈值
            else:
                unique_values = np.unique([np.percentile(dataset[feature], x)
                                           for x in np.linspace(0, 100, 100)])

            # 对可能的分裂阈值求分裂增益，选取增益最大的阈值
            for split_value in unique_values:
                left_targets = targets[dataset[feature] <= split_value]
                right_targets = targets[dataset[feature] > split_value]
                split_gain = self.calc_gini(left_targets['label'], right_targets['label'])

                if split_gain < best_split_gain:
                    best_split_feature = feature
                    best_split_value = split_value
                    best_split_gain = split_gain
        return best_split_feature, best_split_value, best_split_gain

    @staticmethod
    def calc_leaf_value(targets):
        """选择样本中出现次数最多的类别作为叶子节点取值"""
        label_counts = collections.Counter(targets)
        major_label = max(zip(label_counts.values(), label_counts.keys()))
        return major_label[1]

    @staticmethod
    def calc_gini(left_targets, right_targets):
        """分类树采用基尼指数作为指标来选择最优分裂点"""
        split_gain = 0
        for targets in [left_targets, right_targets]:
            gini = 1
            # 统计每个类别有多少样本，然后计算gini
            label_counts = collections.Counter(targets)
            for key in label_counts:
                prob = label_counts[key] * 1.0 / len(targets)
                gini -= prob ** 2
            split_gain += len(targets) * 1.0 / (len(left_targets) + len(right_targets)) * gini
        return split_gain

    @staticmethod
    def split_dataset(dataset, targets, split_feature, split_value):
        """根据特征和阈值将样本划分成左右两份，左边小于等于阈值，右边大于阈值"""
        left_dataset = dataset[dataset[split_feature] <= split_value]
        left_targets = targets[dataset[split_feature] <= split_value]
        right_dataset = dataset[dataset[split_feature] > split_value]
        right_targets = targets[dataset[split_feature] > split_value]
        return left_dataset, right_dataset, left_targets, right_targets

    def predict(self, dataset):
        """输入样本，预测所属类别"""
        res = []
        for _, row in dataset.iterrows():
            pred_list = []
            # 统计每棵树的预测结果，选取出现次数最多的结果作为最终类别
            for tree in self.trees:
                pred_list.append(tree.calc_predict_value(row))

            pred_label_counts = collections.Counter(pred_list)
            pred_label = max(zip(pred_label_counts.values(), pred_label_counts.keys()))
            res.append(pred_label[1])
        return np.array(res)


if __name__ == '__main__':
    '''
    df = pd.read_csv("source/wine.txt")
    df = df[df['label'].isin([1, 2])].sample(frac=1, random_state=66).reset_index(drop=True)
    clf = RandomForestClassifier(n_estimators=5,
                                 max_depth=5,
                                 min_samples_split=6,
                                 min_samples_leaf=2,
                                 min_split_gain=0.0,
                                 colsample_bytree="sqrt",
                                 subsample=0.8,
                                 random_state=66)
    train_count = int(0.7 * len(df))
    feature_list = ["Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols", 
                    "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", 
                    "OD280/OD315 of diluted wines", "Proline"]
    
    clf.fit(df.loc[:train_count, feature_list], df.loc[:train_count, 'label'])

    from sklearn import metrics
    print(metrics.accuracy_score(df.loc[:train_count, 'label'], clf.predict(df.loc[:train_count, feature_list])))
    print(metrics.accuracy_score(df.loc[train_count:, 'label'], clf.predict(df.loc[train_count:, feature_list])))
    '''


    projectList = ['Ant','jruby','kafka','mockito','storm','tomcat']
    # 获取 类似的数据
    '''
    Alcohol,Malic acid,Ash,Alcalinity of ash,Magnesium,Total phenols,Flavanoids,Nonflavanoid phenols,Proanthocyanins,Color intensity,Hue,OD280/OD315 of diluted wines,Proline,label
    14.23,1.71,2.43,15.6,127,2.8,3.06,0.28,2.29,5.64,1.04,3.92,1065,1

    NOF	NOPF	NOM	NOPM	LOC	WMC	NC	DIT	LCOM	FANIN	FANOUT label
    0	0	2	2	7	2	0	0	-1	0	0                        1
    1	0	1	1	6	1	0	0	0	0	0                        2
    1	0	1	1	7	1	0	0	0	0	0                        2
    '''
    import os
    import json
    def Undersampling(trainlist):
        print('trainlist',len(trainlist))
        random.shuffle(trainlist)
        pos = 0
        neg = 0
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
            selectSamples = negSamples + posSamples[:neg]
        elif neg>pos:
            selectSamples = negSamples[:pos] + posSamples
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
    def getTrainOrTestCSV(datalist, csv_txt, jsonFolderPathList):
        label_dict = {}
        for line in datalist:
            name = line.split('    ')[0][:-5]+'.json'
            label = line.split('    ')[1]
            #print('label',label)
            label_dict[name] = label
        csv_data = []
        csv_data.append('LOC,CC,PC,label\n')
        for jsonFolderPath in jsonFolderPathList:
            for root ,dirs, files in os.walk(jsonFolderPath):
                for file in files:
                    if file in label_dict and file.endswith('.json'):
                        data = json.load(open(root+'/'+file))
                        src_metrics = data["metrics"]
                        label = 1 if label_dict[file] == '1' else 2
                        
                        csv_data.append(','.join(map(lambda x:str(x), src_metrics))+','+str(label)+'\n')
        data_txt = open(csv_txt,'w')
        data_txt.writelines(csv_data)
        data_txt.close()

    def get_10_fold_cross_train_test_list(project):
        trainlist = []
        testlist = []
        for projectname in projectList:
            if projectname != project:
                train_list_path = "/home/yqx/Downloads/DesigniteJava-master/myData/"+projectname+"/methods/trainlabels.txt"
                trainlist += open(train_list_path).readlines()
                train_list_path = "/home/yqx/Downloads/DesigniteJava-master/myData/"+projectname+"/methods/testlabels.txt"
                trainlist += open(train_list_path).readlines()
            else:
                test_list_path = "/home/yqx/Downloads/DesigniteJava-master/myData/"+projectname+"/methods/trainlabels.txt"
                testlist += open(test_list_path).readlines()
                test_list_path = "/home/yqx/Downloads/DesigniteJava-master/myData/"+projectname+"/methods/testlabels.txt"
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

    record_file = open('result_method_6_cross_1.txt', 'a')
    for project in projectList:
        jsonFolderPathList_train = []
        jsonFolderPathList_test = []
        for projectname in projectList:
            if projectname != project:
                jsonFolderPathList_train.append("/home/yqx/Downloads/DesigniteJava-master/myData/"+projectname+"/methods/rawjson")
            else:
                jsonFolderPathList_test.append("/home/yqx/Downloads/DesigniteJava-master/myData/"+projectname+"/methods/rawjson")

        trainlist,testlist = get_10_fold_cross_train_test_list(project)
        trainlist = Undersampling(trainlist)
        #print("trainlist",trainlist)
        getTrainOrTestCSV(trainlist, 'train_csv.txt', jsonFolderPathList_train)
        getTrainOrTestCSV(testlist, 'test_csv.txt', jsonFolderPathList_test)

        ##################################################################################################
        train_data = pd.read_csv("train_csv.txt")
        train_data = train_data[train_data['label'].isin([1, 2])].sample(frac=1, random_state=66).reset_index(drop=True)

        test_data = pd.read_csv("test_csv.txt")
        test_data = test_data[test_data['label'].isin([1, 2])].sample(frac=1, random_state=66).reset_index(drop=True)

        clf = RandomForestClassifier(n_estimators=128,
                                    max_depth=16,
                                    min_samples_split=2,
                                    min_samples_leaf=2,
                                    min_split_gain=0.0,
                                    colsample_bytree="sqrt",
                                    subsample=0.8,
                                    random_state=66)
        
        feature_list = ['LOC','CC','PC']
        clf.fit(train_data.loc[:, feature_list], train_data.loc[:, 'label'])
        
        from sklearn import metrics
        print("project:",project)
        #print("train_accuracy_score", metrics.accuracy_score(train_data.loc[:, 'label'], clf.predict(train_data.loc[:, feature_list])))
        #print("test_accuracy_score", metrics.accuracy_score(test_data.loc[:, 'label'], clf.predict(test_data.loc[:, feature_list])))
        #print("test_precision_score", metrics.precision_score(test_data.loc[:, 'label'], clf.predict(test_data.loc[:, feature_list])))
        #print("test_recall_score", metrics.recall_score(test_data.loc[:, 'label'], clf.predict(test_data.loc[:, feature_list])))
        #print("test_f1_score", metrics.f1_score(test_data.loc[:, 'label'], clf.predict(test_data.loc[:, feature_list])))
        #print('\n')

        record_file.write("project:"+ project+'\n')
        record_file.write("train_accuracy_score"+ str(metrics.accuracy_score(train_data.loc[:, 'label'], clf.predict(train_data.loc[:, feature_list])))+'\n')
        record_file.write("test_accuracy_score"+ str(metrics.accuracy_score(test_data.loc[:, 'label'], clf.predict(test_data.loc[:, feature_list])))+'\n')
        record_file.write("test_precision_score"+ str(metrics.precision_score(test_data.loc[:, 'label'], clf.predict(test_data.loc[:, feature_list])))+'\n')
        record_file.write("test_recall_score"+ str(metrics.recall_score(test_data.loc[:, 'label'], clf.predict(test_data.loc[:, feature_list])))+'\n')
        record_file.write("test_f1_score"+ str(metrics.f1_score(test_data.loc[:, 'label'], clf.predict(test_data.loc[:, feature_list])))+'\n\n\n')
    record_file.close()