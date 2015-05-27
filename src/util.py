#!/usr/local/op_paas/python2.7/bin/python
# -*- coding: gbk -*-

import math
import re

def gbk2unicode(s_gbk):
    return s_gbk.decode('gbk')

def strQ2B(ustring):
    rstring = ''
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif inside_code >= 65281 and inside_code <= 65374:
            inside_code -= 65248

        rstring += unichr(inside_code)
    return rstring

def strB2Q(ustring):
    '''
    半角转全角.
    '''
    rstring = ''
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 32:
            inside_code = 12288
        elif inside_code >= 32 and inside_code <= 126:
            inside_code += 65248

        rstring += unichr(inside_code)
    return rstring

def hamming_dist(s1, s2):
    if len(s1) != len(s2):
        raise ValueError("Undefined for sequences of unequal length")
    return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))

def edit_dist(s1, s2):
    l1 = len(s1)
    l2 = len(s2)
    if l1 == 0:
        return l2
    if l2 == 0:
        return l1
    if l1 > l2:
        s1, s2 = s2, s1
        l1, l2 = l2, l1

    mat = [range(l2) if x==0 else [x if y==0 else 0 for y in range(l2)] for x in range(l1)]
    for i in range(1, l1):
        for j in range(1, l2):
            mat[i][j] = min(mat[i-1][j] + 1, mat[i][j-1] + 1, mat[i-1][j-1]+(0 if s1[i-1]==s2[j-1] else 1))
    return mat[l1-1][l2-1]

def lcseq(s1, s2):
    m = len(s1)
    n = len(s2)
    L = [[0 for j in range(n+1)] for i in range(m+1)]
    for i in range(m, -1, -1):
        for j in range(n, -1, -1):
            if i == m or j == n:
                L[i][j] = 0
            elif s1[i] == s2[j]:
                L[i][j] = 1 + L[i+1][j+1]
            else:
                L[i][j] = max(L[i+1][j], L[i][j+1])

    return L[0][0]

def lcstr(s1, s2):
    rs = 0
    m = len(s1)
    n = len(s2)
    mat = [[0 for j in range(n)] for i in range(m)]
    for i in range(m):
        for j in range(n):
            if s1[i] == s2[j]:
                mat[i][j] = 1 + (0 if i==0 or j==0 else mat[i-1][j-1])
                if mat[i][j] > rs:
                    rs = mat[i][j]
            else:
                mat[i][j] = 0
    return rs

def cos_sim(word_dict1, word_dict2):
    if len(word_dict1) == 0 or len(word_dict2) == 0:
        return 0.0
    if len(word_dict1) > len(word_dict2):
        word_dict1, word_dict2 = word_dict2, word_dict1
    num1 = 0.0
    num2 = 0.0
    dot_product = 0.0
    for w, val in word_dict1.items():
        num1 += val**2
        if w in word_dict2:
            dot_product += val * word_dict2[w]

    num2 = sum(val**2 for w, val in word_dict2.items())
    return dot_product / (math.sqrt(num1) * math.sqrt(num2))

def cos_sim_tm(tm_list1, tm_list2):
    dot_product = 0.0
    num1 = 0.0
    num2 = 0.0
    dict1 = {}
    dict2 = {}
    for i, val in tm_list1:
        dict1[i] = val
        num1 += val * val
    for i, val in tm_list2:
        dict2[i] = val
        num2 += val * val
    for w, val in dict1.items():
        if w in dict2:
            dot_product += val * dict2[w]

    if(num1 == 0.0 or num2 ==0.0):
        # print 'num1 or num2 == 0 !!!'
        return 0
    return dot_product / (math.sqrt(num1) * math.sqrt(num2))

def jaccard_sim(word_set1, word_set2):
    if len(word_set1) == 0 or len(word_set2) == 0:
        return 0.0
    return float(len(word_set1.intersection(word_set2))) / len(word_set1.union(word_set2))

def del_sub_title(title):
    '''
    Delete sub title, title DBC, eg: title_subtile => title, title - subtitle => title.
    '''
    symbols = ['-', '_']
    index = -1
    for symbol in symbols:
        i = title.find(symbol)
        if i == 0:
            i = title.find(symbol, 1)
        if index == -1:
            index = i
        else:
            if i != -1 and index > i:
                index = i
    return (title, '') if index == -1 else (title[: index], title[index+1: ])

def eng_to_lower(s):
    return s.lower()

def process_str(s):
    bad_char = r'[。]'
    return re.sub(bad_char, '.', s)

def process_title(t):
    symbol_list = ['《', '》', '？', '【', '】', '（', '）', '“', '”']
    for symbol in symbol_list:
        t = t.replace(symbol, '')
    # print t
    bad_char = r'[\s?\+<>\(\)""]'
    t = re.sub(bad_char, '', t)
    return t

def process_query(q):
    symbol_list = ['【', '】', '？']
    for sysbol in symbol_list:
        q = q.replace(symbol, '')
    return q

digit_dict = {}
def load_digit_dict(filename):
    with open(filename) as f:
        for line in f:
            character_digit, digit = line.rstrip().split(',')
            digit_dict[gbk2unicode(character_digit)] = gbk2unicode(digit)
    # print digit_dict
    print 'Load Digit Dict Complete.'


def digit_norm(s, other_str):
    '''
    s need to be unicode.
    '''
    rs = ''
    pattern = re.compile(u'[零一二三四五六七八九十]+', re.UNICODE)
    start = 0
    for m in pattern.finditer(s):
        # print m.start(), len(m.group())
        if m.group() in digit_dict and digit_dict[m.group()] in other_str:
            rs += s[start : m.start()] + digit_dict[m.group()]
        else:
            rs += s[start : m.start() + len(m.group())]
        start = m.start() + len(m.group())
        # print rs
    rs += s[start:]
    return rs

def get_url_title(s):
    '''
    Is valid url? reconstruct url, return url title.
    '''
    pass

def gen_digit_dict(output='../data/digit_dict', total=3000):
    origin_dict = {0 : '零', 1 : '一',  2 : '二', 3 : '三', 4 : '四', 5 : '五', 6 : '六', 7 : '七', 8 : '八', 9 : '九', 10 : '十'}
    digit_list = []
    for i in range(total):
        if i <= 10:
            digit_list.append( (origin_dict[i], i) )
        elif 10 < i < 100:
            digit_list.append( ( ('' if i / 10 == 1 else origin_dict[i / 10]) + '十' + ('' if i % 10 == 0 else origin_dict[i % 10]), i) )
        elif 100 <= i < 1000:
            digit_list.append( (origin_dict[i / 100] + '百' + ('' if i % 100 == 0 else (('零'+origin_dict[i % 100]) if i % 100 < 10 else (origin_dict[i % 100 / 10] + '十' + ('' if i % 100 % 10 == 0 else origin_dict[i % 100 % 10]) ))), i ) )
        elif 1000 < i < 3000:
            digit_list.append( ((origin_dict[i / 1000] + origin_dict[i % 1000 / 100] + origin_dict[i % 100 / 10] + origin_dict[i % 10]), i ) )
            if i == 1001:
                digit_list.append( ('一千零一', i) )

    with open(output, 'w') as f:
        for character, digit in digit_list:
            # print character, digit
            f.write(character + ',' + str(digit) + '\n')

qc_dict = {}
with open('../data/qc.txt') as f:
    for line in f:
        query, rs = line.rstrip().split(',')
        qc_dict[gbk2unicode(query)] = gbk2unicode(rs)
def qc(q):
    if q in qc_dict:
        q = qc_dict[q]
    return q

def main():
    print strQ2B('A（【？《“'.decode('gbk'))

    print 'Edit Dist: ' + str(edit_dist('宝马x5', 'x5宝马'))
    print lcseq('abcd', '')
    print lcstr('abcfe1234erg', '')
    d1 = {1: 0, 2: 0.1}
    d2 = {2: 2, 3: 1}
    print cos_sim(d1, d2)
    set1 = set([1, 2, 3, 4])
    set2 = set([3, 2, 5, 6, 7])
    print jaccard_sim(set1, set2)
    print del_sub_title('ab-c - _subtitle - subtitle _subtitle')
    print eng_to_lower('我的bacAbdefaDT$%')
    t = '搜狗  《短文本》【相关性】（比赛）(加油)“呵呵”"嘿嘿"'
    print 'Title: ' + t
    print process_title(t)
    
    print digit_norm(u'佳域g6是一体机么', u'1体机')
    # gen_digit_dict()

if __name__ == '__main__':
    main()

