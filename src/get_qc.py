#! -*- coding: utf-8 -*-

import sys
import requests
from pyquery import PyQuery as pq

def gen_qc_file(query_file, qc_file):
    with open(qc_file, 'w') as fout:
        with open(query_file) as fin:
            for line in fin:
                query = line.rstrip()
                qc = get_qc(query)
                if len(qc) > 0:
                    try:
                        fout.write(query + ',' + qc.encode('gbk') + '\n')
                    except Exception, e:
                        print e
                        print query
                        # fout.write(query + ',' + qc + '\n')

HEADERS = {'User-Agent': 'Mozilla/5.0'}

def get_qc(query):
    qc = ''
    try:
        url = r'http://www.baidu.com/s?wd='
        print url + query
        res = requests.get(url + query, headers=HEADERS, timeout=1)
        qc = pq(res.content).find('div').filter('.hit_top_new')('a')('strong').html()
        if qc == None:
            qc = pq(res.content).find('div').filter('.hit_top_new')('span')('strong').html()
        if qc == None:
            res = requests.get(r'http://www.sogou.com/web?ie=utf-8&query=' + query, headers=HEADERS, timeout=1)
            qc = pq(res.content).find('div').filter('.topqc')('strong')('em').html()
            # print qc[1]   # codec problem ... 
        if qc == None:
            qc = ''
        
    except Exception, e:
        print e
        print query
        pass
    return qc

def main():
    q = r'shijiebei'
    q = r'ΑυµΓ»ª'
    get_qc(q)
    if len(sys.argv) != 3:
        print 'Usage: python27 get_qc.py querys_test.txt qc.txt'
    else:
        gen_qc_file(sys.argv[1], sys.argv[2])

if __name__ == '__main__':
    main()
