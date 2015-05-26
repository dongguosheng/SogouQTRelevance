#!/usr/local/op_paas/python2.7/bin/python
# -*- coding: utf-8 -*-

import random
import sys

def split_train_test(input_file='data.txt', train_set='train_off.txt', test_set='test_off.txt'):
    '''
    Train Set : Test Set = 2 : 1
    '''
    query_title_dict = {}
    with open(input_file) as f:
        for line in f:
            query, title_rate = line.rstrip().split('\t', 1)
            if query not in query_title_dict:
                query_title_dict[query] = []
            query_title_dict[query].append(title_rate)

    query_title_list = [(k, v) for k, v in query_title_dict.items()]
    random.shuffle(query_title_list)

    two_third = len(query_title_list) / 3 * 2
    print 'two_third: %d' % two_third
    f_train = open(train_set, 'w')
    f_test = open(test_set, 'w')
    for i in range(len(query_title_list)):
        if i < two_third:
            for title_rate in query_title_list[i][1]:
                f_train.write(query_title_list[i][0] + '\t' + title_rate + '\n')
        else:
            for title_rate in query_title_list[i][1]:
                f_test.write(query_title_list[i][0] + '\t' + title_rate + '\n')

    f_train.close()
    f_test.close()

def main():
    if len(sys.argv) != 4:
        print 'Usage: python split_train_test.py data.txt train.txt test.txt'
    else:
        split_train_test(input_file=sys.argv[1], train_set=sys.argv[2], test_set=sys.argv[3])

if __name__ == '__main__':
    main()
