import json
import javalang
import pandas as pd
import util
from gensim.models.word2vec import Word2Vec
from collections import Counter

from tqdm import tqdm
import regex as re
import os
import sys
sys.setrecursionlimit(1000000)

def java_method_to_ast(method):
    tokens = javalang.tokenizer.tokenize(method)
    parser = javalang.parser.Parser(tokens)
    tree = parser.parse_member_declaration()
    return tree

def preprocess_data(projectname):
    print('Reading all code to code_list...')
    #读取所有label到dict
    print("projectname",projectname)

    train_labels = "/home/yqx/Downloads/DesigniteJava-master/myData/"+projectname+"/types/trainlabels.txt"
    test_labels = "/home/yqx/Downloads/DesigniteJava-master/myData/"+projectname+"/types/testlabels.txt"
    typefilepath = "/home/yqx/Downloads/DesigniteJava-master/myData/"+projectname+"/types/pureJava"

    label_dict = {}
    train_label_file = open(train_labels)
    #tomcat-10.0.0-org.apache.catalina.session-ManagerBase.java    0
    for line in train_label_file:
        label_dict[line.split('    ')[0]] = int(line.split('    ')[1])
    test_label_file = open(test_labels)
    for line in test_label_file:
        label_dict[line.split('    ')[0]] = int(line.split('    ')[1])

    for root, ds, files in os.walk(typefilepath):
        name_list = []
        ast_list = []
        label_list = []
        parse_error = 0
        for file in tqdm(files):
            if file not in label_dict:
                print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',file)
                continue
            codepath = root + '/' + file
            if os.path.getsize(codepath) > 1024*50:
                print("文件大于50kb")
                continue
            codefile=open(codepath,encoding='utf-8')
            programtext=codefile.read()
            lines = codefile.readlines()
            
            programtext = re.sub(r'((\/[*]([*].+|[\\n]|\\w|\\d|\\s|[^\\x00-\\xff])+[*]\/))', "", programtext)
            codefile.close()
            try:
                tokens = javalang.tokenizer.tokenize(programtext)
                parser = javalang.parser.Parser(tokens)
                tree = parser.parse_member_declaration()
            except:
                print('parse error')
                parse_error+=1
                continue
            name_list.append(file)
            ast_list.append(tree)
            label_list.append(label_dict[file])
    print("parse_error",parse_error)   
    data = {"name":name_list,"ast":ast_list,"label":label_list}
    programs = pd.DataFrame(data,columns = ['name','ast','label'])

    print('Data size:', len(programs))

    print('Training word embedding...')

    programs['corpus'] = programs['ast'].apply(util.ast2sequence)

    #print("programs",programs)
    w2v = Word2Vec(programs['corpus'], size=128, workers=16, sg=1, min_count=3)  # use w2v[WORD] to get embedding
    vocab = w2v.wv.vocab

    # transform ASTNode to tree of index in word2vec model
    def node_to_index(node):
        result = [vocab[node.token].index if node.token in vocab else len(vocab)]
        for child in node.children:
            result.append(node_to_index(child))
        return result


    # transform ast to trees of index in word2vec model
    def ast_to_index(ast):
        blocks = []
        util.get_ast_nodes(ast, blocks)
        return [node_to_index(b) for b in blocks]

    print('Transforming ast to embedding index tree...')

    programs['index_tree'] = programs['ast'].apply(ast_to_index)

    #print("programs['code']",programs['code'])
    #print("programs['ast']",programs['ast'])
    #exit()
    programs.pop('ast')
    programs.pop('corpus')
    #print("programs",programs)

    w2v.save('./data/w2v_128')
    print("Saved word2vec model at", './data/w2v_128')
    programs.to_pickle('./data/programs.pkl')
    print("Saved processed data at", './data/programs.pkl')







