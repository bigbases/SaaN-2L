from utils import loadData
import sys
import random
import math
import datetime
import os
import subprocess
import threading
import time
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
from numpy import sum as np_sum, dot, transpose, array, mean, var, std
# from gaussEmbed import GaussEmbeddingNetwork
from utils.transform import adjList_to_edgeList, edgeList_to_adjList
from copy import deepcopy
from utils.train_test import gen_train_test, gen_train_test_from_walks
from sklearn.metrics import f1_score

def test(embedding, label, input_file):
    classify_input = input_file+'_classify.txt'
    classify_input_file = open(classify_input, 'w')
    for k in label:
        classify_input_file.write(str(label[k]) + ' ')
        for feature in range(len(embedding[k])):
            classify_input_file.write(str(feature+1) + ':' + str(embedding[k][feature]) + ' ')
        classify_input_file.write('\n')
    classify_input_file.close()
    # F1_test(classify_input)
    svm_command = './train -s 0 -v 10 ' + classify_input
    subprocess.call(svm_command, shell=True)

def multi_test(embedding, label, input_file, train_ratio=0.9):
    verts = [key for key in label]
    # verts = []
    # for key in label:
    #     if embedding[key] != []:
    #         verts.append(key)
    # print verts
    result = []
    for i in range(10):
        random.shuffle(verts)
        train_verts = verts[0:int(len(verts)*train_ratio)]
        test_verts = verts[int(len(verts)*train_ratio):]

        train_file = open(input_file+str(train_ratio)+'_train.txt', 'w')
        test_file = open(input_file+str(train_ratio)+'_test.txt', 'w')

        for vert in train_verts:
            for one_embedding in embedding[vert]:
                train_file.write(str(label[vert])+' ')
                for feature in range(len(one_embedding)):
                    train_file.write(str(feature+1)+':'+str(one_embedding[feature])+' ')
                train_file.write('\n')
        train_file.close()

        line_count = 0
        line_index_dict = {}
        for vert in test_verts:
            if embedding[vert] == []:
                print(vert)
            for one_embedding in embedding[vert]:
                test_file.write(str(label[vert])+' ')
                for feature in range(len(one_embedding)):
                    test_file.write(str(feature+1)+':'+str(one_embedding[feature])+' ')
                test_file.write('\n')
                if vert in line_index_dict:
                    line_index_dict[vert].append(line_count)
                else:
                    line_index_dict[vert] = [line_count]
                line_count += 1
        test_file.close()
        # print line_index_dict

        train_command = './train -s 0 ' + input_file+str(train_ratio)+'_train.txt ' + input_file + 'model.txt' 
        subprocess.call(train_command, shell=True)
        predict_command = './predict -b 1 ' + input_file+str(train_ratio)+'_test.txt ' + \
            input_file + 'model.txt ' + input_file + 'output.txt' 
        subprocess.call(predict_command, shell=True)

        predict_file = open(input_file + 'output.txt')
        line0 = predict_file.readline()
        line0arr = line0.strip().split()
        label_list = [int(k) for k in line0arr[1:]]

        # print line_index_dict[712]

        prob_list = []
        for line in predict_file.readlines():
            lineArr = line.strip().split()
            prob_list.append( [float(k) for k in lineArr[1:]])
        # print prob_list

        y_true = []
        y_pred = []
        y_pred2 = []
        for vert in test_verts:
            if vert not in line_index_dict:
                continue
            y_true.append(label[vert])
            avg_prob_list = []
            for i in range(len(label_list)):
                total = 0.0
                for preds in line_index_dict[vert]:
                    total += prob_list[preds][i]
                avg_prob_list.append(total/len(line_index_dict[vert]))
            max_label_index = 0
            max_prob = avg_prob_list[0]
            for i in range(len(avg_prob_list)):
                if avg_prob_list[i] > max_prob:
                    max_label_index = i
                    max_prob = avg_prob_list[i]
            y_pred.append(label_list[max_label_index])

            max_prob2 = prob_list[line_index_dict[vert][0]][0]
            max_label_index2 = 0
            for i in range(len(label_list)):
                for preds in line_index_dict[vert]:
                    if prob_list[preds][i] > max_prob2:
                        max_prob2 = prob_list[preds][i]
                        max_label_index2 = i
            y_pred2.append(label_list[max_label_index2])

        right = 0.0
        for i in range(len(y_true)):
            if y_true[i] == y_pred[i]:
                right += 1
        print(right/len(y_true))
        print('len(y_true)', len(y_true))
        print('len(test_verts)', len(test_verts))
        # micro_f1 = f1_score(y_true, y_pred, average='micro')
        # macro_f1 = f1_score(y_true, y_pred, average='macro')
        # print micro_f1, macro_f1    
        result.append(right/len(y_true))
        # micro_f1 = f1_score(y_true, y_pred2, average='micro')
        # macro_f1 = f1_score(y_true, y_pred2, average='macro')
        # print micro_f1, macro_f1
        # result.append(micro_f1)
    print('result', result )
    result_sum = sum(result)
    classify_input = input_file+'_classify.txt'
    classify_input_file = open(classify_input, 'w')
    for k in label:
        for one_embedding in embedding[k]:
            classify_input_file.write(str(label[k]) + ' ')
            for feature in range(len(one_embedding)):
                classify_input_file.write(str(feature+1) + ':' + str(one_embedding[feature]) + ' ')
            classify_input_file.write('\n')
    # F1_test(classify_input)
    svm_command = './train -s 0 -v 10 ' + classify_input
    # subprocess.call(svm_command, shell=True)
    print('avg', result_sum/len(result))
    print(mean(result), 1.96*std(result) )
    # classify_input = input_file+'_classify.txt'
    # classify_input_file = open(classify_input, 'w')
    # for k in label:
    #     for one_embedding in embedding[k]:
    #         classify_input_file.write(str(label[k]) + ' ')
    #         for feature in range(len(one_embedding)):
    #             classify_input_file.write(str(feature+1) + ':' + str(one_embedding[feature]) + ' ')
    #         classify_input_file.write('\n')
    # classify_input_file.close()
    # # F1_test(classify_input)
    # svm_command = './train -v 10 ' + classify_input
    # subprocess.call(svm_command, shell=True)  

def F1_test(classify_input):
    f = open(classify_input)
    lines = f.readlines()
    random.shuffle(lines)
    train_ratio_list = [0.9]
    for train_ratio in train_ratio_list:
        line_train = lines[0:int(train_ratio*len(lines))]
        line_test = lines[int(train_ratio*len(lines)):]
        f_train = open(classify_input+'_train'+str(train_ratio), 'w')
        f_test = open(classify_input+'_test'+str(train_ratio), 'w')
        f_train.writelines(line_train)
        f_test.writelines(line_test)
        f_train.close()
        f_test.close()
    f.close()


def main():
    random.seed(1)
    parser = ArgumentParser(description = 'graph representation learning')
    parser.add_argument('--input', required=True, help='Name of edgelist file.')
    parser.add_argument('--label', required=True, help='groups file')
    parser.add_argument('--veclen', default=64, type = int, help = 'Length of the vertex vector')
    parser.add_argument('--reset', type=int, required=True)
    parser.add_argument('--number-walks', default = 10, type = int, help = 'Number of random walks to start at each node')
    parser.add_argument('--walk-length', default = 40, type = int, help = 'Length of the random walk started at each node')
    parser.add_argument('--window-size', default = 5, type = int, help = 'Window size of skipgram model.')
    parser.add_argument('--workers', default=1, type=int, help='')
    parser.add_argument('--metric', type=int, required=True)
    parser.add_argument('--gauss', type=int, default=0)
    parser.add_argument('--mssg', type=int, default=0)
    parser.add_argument('--trainMethod', type=int, default=0)
    parser.add_argument('--line', type=int, default=1)
    parser.add_argument('--covm', default = 0.5, type = float, help = 'minimum covariance')
    parser.add_argument('--covM', default = 1, type = float, help = 'maximum covariance')
    parser.add_argument('--margin', default=10, type=float, help='margin')
    parser.add_argument('--cmty', default=100, type=int, help='cmty numbers')
    parser.add_argument('--train-ratio', default=0.9, type=float, help='train ratio')
    #node2vec
    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    args = parser.parse_args()
    label = loadData.load_label(args.label)
    max_index = 0
    for k in label:
        if k > max_index:
            max_index = k
    edgelist = loadData.loadEdgeList(args.input)
    adjlist = edgeList_to_adjList(edgelist, max_index+1)

    logfile = open(args.input+'_log', 'w')
    logfile.write('metric'+str(args.metric)+'gauss'+str(args.gauss)+'mssg'+str(args.mssg))
    logfile.close()
    # 0:original dw, 2:gauss dw, 3:mssg
    if args.metric == 0 or args.metric == 3:
        # train embeddings
        output = args.input + '_embed.txt'
        dw_command = '/home/zengxiangkai/software/python27/bin/python deepwalk/__main__.py --format edgelist --input ' + args.input + \
            ' --window-size ' + str(args.window_size) + ' --walk-length ' + str(args.walk_length) + ' --number-walks ' + str(args.number_walks) + \
            ' --output ' + output + ' --gauss ' + str(args.gauss) + ' --mssg ' + str(args.mssg) + \
            ' --trainMethod ' + str(args.trainMethod) + ' --representation-size ' + str(args.veclen) + \
            ' --covm ' + str(args.covm) + ' --covM ' + str(args.covM) + ' --workers ' + str(args.workers)+ \
            ' --p ' + str(args.p) + ' --q ' + str(args.q)
        if args.reset == 1:
            subprocess.call(dw_command, shell=True)
        embedding, veclen = loadData.loadMultiSenseVec(output, max_index+1)
        multi_test(embedding, label, args.input, args.train_ratio)


    # lda
    if args.metric == 7:
        output = args.input + '_embed.txt'
        dw_command = '/home/zengxiangkai/software/python27/bin/python deepwalk/__main__.py --format edgelist --input ' + args.input + \
        ' --window-size ' + str(args.window_size) + ' --walk-length ' + str(args.walk_length) + ' --number-walks ' + str(args.number_walks) + \
        ' --output ' + output + ' --gauss ' + str(args.gauss) + ' --mssg ' + str(args.mssg) + \
        ' --trainMethod ' + str(args.trainMethod) + ' --representation-size ' + str(args.veclen) + \
        ' --covm ' + str(args.covm) + ' --covM ' + str(args.covM) + ' --CNE 3' + ' --cmty-num ' + str(args.cmty) + ' --maxIndex ' + str(max_index+1) + ' --workers ' + str(args.workers)+ \
        ' --p ' + str(args.p) + ' --q ' + str(args.q)
        if args.reset == 1:
            subprocess.call(dw_command, shell=True)
        embedding, veclen = loadData.loadMultiSenseVec(output, max_index+1)
        multi_test(embedding, label, args.input, args.train_ratio)  

        # embedding, veclen = loadData.loadMultiSenseVec(output+'_multi', max_index+1)
        # multi_test(embedding, label, args.input, args.train_ratio)   
    # half Lda embed
    if args.metric == 8:
        output = args.input + '_embed.txt'
        dw_command = '/home/zengxiangkai/software/python27/bin/python deepwalk/__main__.py --format edgelist --input ' + args.input + \
        ' --window-size ' + str(args.window_size) + ' --walk-length ' + str(args.walk_length) + ' --number-walks ' + str(args.number_walks) + \
        ' --output ' + output + ' --gauss ' + str(args.gauss) + ' --mssg ' + str(args.mssg) + \
        ' --trainMethod ' + str(args.trainMethod) + ' --representation-size ' + str(args.veclen) + \
        ' --covm ' + str(args.covm) + ' --covM ' + str(args.covM) + ' --CNE 4' + ' --cmty-num ' + str(args.cmty) + ' --maxIndex ' + str(max_index+1) + ' --workers ' + str(args.workers)+ \
        ' --p ' + str(args.p) + ' --q ' + str(args.q)
        if args.reset == 1:
            subprocess.call(dw_command, shell=True)
        embedding, veclen = loadData.loadMultiSenseVec(output, max_index+1)
        multi_test(embedding, label, args.input, args.train_ratio)  




if __name__ == '__main__':
    main()
