# -*- coding: utf-8 -*-
#改进版，前几版使用的是欧式距离，此版开始使用余弦距离
import jieba
import jieba.analyse
import math
import numpy as np
from numpy import *
import time
#import matplotlib.pyplot as plt

#set_printoptions(threshold='nan')
#start_time = datetime.datetime.now()
board_list=['Basketball','D_Computer','V_Suggestions','Mobile','JobExpress',
            'Stock','WarAndPeace','Girls','FleaMarket','WorldFootball']
file_list = ["lily/Basketball.txt","lily/D_Computer.txt","lily/V_Suggestions.txt",
             "lily/Mobile.txt","lily/JobExpress.txt","lily/Stock.txt",
             "lily/WarAndPeace.txt","lily/Girls.txt","lily/FleaMarket.txt",
             "lily/WorldFootball.txt"]
stopword = set(jieba.cut(open("lily/stopwords.txt").read()))  #stopwords集
N = 5  #距离的N
K = 30 #K个距离


######################
#关键词去重
#key_set = []
#for item in file_list:
#    f = open(item).read()
#    key_set.append( set(jieba.analyse.extract_tags(f,300)) )
#    
#for i in range( 0 , len(key_set) ):
#    for j in range( i+1 , len(key_set) ):
#        section = key_set[i].intersection( key_set[j] )
#        stopword.union(section)
#    
#
#del key_set
######################
#收集关键词
key_dict = {}

for item in file_list:
##    print item,"->收集关键词"
    f = open(item).read()
    keys = jieba.analyse.extract_tags(f,300)
    count = 0
    for word in keys:
        if (word in stopword) or word.isdigit() or (key_dict.has_key(word)):
##            print "==============stop word =============",word
            continue
        else:
            count += 1
            key_dict[word] = 0
##            print word
        if count == 150:
            
            break
        
#
#END OF "收集关键词"
#######################

#######################
#构建矩阵
#
#

tf_list_class = [] #此list包含所有类所有帖子的tf，每个元素是一个二级列表，
                #存放此类中所有帖子的TF。二级列表的每个元素为一个三级
                #列表，存放每帖子每个关键词的TF

class_index = 0
for File in file_list:
    #构建TFIDF矩阵，粒度为帖子
    
    topics = open(File).readlines()
    for item in topics:     #对于每个帖子
    
        #计数词典清零
        temp_topicTF_list = [class_index]  
        for element in key_dict:   #词典用于辅助计数，此处在开始统计某词在某类出现频率前清零词典计数
            key_dict[element] = 0
        cutted = list(jieba.cut(item))
        sum_words_in_this_topic = float( len(cutted) ) #此帖子中的总词数
        for word in cutted:  #计算每个关键词，计算出现次数
            if key_dict.has_key(word):
                key_dict[word] += 1
        for element in key_dict:    #计算此帖子内，每个关键词的TF-IDF
            temp_topicTF_list.append(key_dict[element]/sum_words_in_this_topic)
        tf_list_class.append( temp_topicTF_list )

    class_index += 1
    #TFIDF词典构建完毕

tf_list_class = np.array( tf_list_class )

#
#
#矩阵构建完成，产出tf_list_calss列表
########################
#judge()
#
#

def distance( x , train_vector ):
    x = np.array(x)
    train_vector = np.array(train_vector)
    temp_vector = np.dot( x , train_vector)/(module( x ) * module( train_vector ))

    return temp_vector

#求向量的模
def module( x ):
    tmp = 0
    for item in x:
        tmp += item **2
    return tmp ** (0.5)

#对x分类，结果返回x的类序号
precision = []
time_l = []
def classify( Class , x , tf_list_class = tf_list_class ):
    list_distance = []
    list_class = []
    start_time = time.time()
    
    #得到与训练集中所有元素的距离列表
    for topic in tf_list_class:
        list_class.append(topic[0])
        list_distance.append( distance( x , topic[1:] ) )

    #得到距离最小的 K个 距离类 
    classify_list = []

    for top_k in range( 0 , K):
        index = list_distance.index( max(list_distance) )
        classify_list.append( list_class[index] )
        del list_distance[index]
        del list_class[index]
  
    #得到投票最多的类
    vote_list = [0]*len(board_list)
    for item in classify_list:
        vote_list[int(item)] += 1
    
    precision.append(vote_list[ Class ]/float(K)) 
    end_time = time.time()
    time_l.append( end_time - start_time )    
    return vote_list.index( max( vote_list ) )
    
def judge( Class , x ):
    jud = classify( Class , x )
#    print "judge:\t",jud,"\tfact:\t",Class
    if jud == Class:
        return 1
    else:
        return 0
        

#
#
#函数定义完毕
#############################
#测试
#
#
bench0 = 0
bench1 = 0

for Class_index in range(0,len(board_list)):
#    b0 = 0
#    b1 = 0
    f = open(board_list[Class_index]).readlines()
    for item in f:
        #计数词典清零
        temp_topicTF_list = []  # 1 为偏移量 b 对应的向量 x 中的值
        for element in key_dict:   #词典用于辅助计数，此处在开始统计某词在某类出现频率前清零词典计数
            key_dict[element] = 0
        cutted = list(jieba.cut(item))
        sum_words_in_this_topic = float( len(cutted) ) #此帖子中的总词数
        for word in cutted:  #计算每个关键词，计算出现次数
            if key_dict.has_key(word):
                key_dict[word] += 1
                
        for element in key_dict:    #计算此帖子内，每个关键词的TF-IDF
            temp_topicTF_list.append(key_dict[element]/sum_words_in_this_topic)
        
        bench = judge( Class_index , temp_topicTF_list )
#        if 1 == bench:
#            bench1 += 1
##            b1 += 1
#        elif 0 == bench:
#            bench0 += 1
##            b0 += 1
#        else:
#            print "error"
#            exit(1)
            
#    print "b0:\t",b0,"\tb1:\t",b1
#    print float(b1)/(b0 + b1)

#print "bench0 :",bench0
#print "bench1 :",bench1            
#print float(bench1)/(bench1+bench0)
#print precision
def standard_deviation( mean , precision = precision ):
    sum_d = 0
    for item in precision:
        sum_d += (item - mean ) **2
    
    return (sum_d / len(precision)) **0.5
    
mean_pre = sum(precision) / len(precision)
print "mean precision : ",mean_pre,
print "standard deviation : ",standard_deviation( mean_pre )
#end_time = datetime.datetime.now()
#print "cost : ",(end_time - start_time).seconds , " s"
mean_time = sum(time_l) / len(time_l)
print "mean cost : ",mean_time , " s"
print "standard deviation : ",standard_deviation( mean_time )
#
#
#
#############################