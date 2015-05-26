#!/bin/python
# -*- coding: gbk -*-

import requests
from pyquery import PyQuery as pq
import time
import sys

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko',
            'Accept-Language' : 'zh-CN,zh;q=0.8,en;q=0.6',
            'Accept-Encoding' : 'gzip,deflate'}
           # 'Host' : 'www.baidu.com',
           # 'Connection' : 'Keep-Alive',
           # 'Accept' : 'text/html,application/xhtml+xml,*/*'}

def get_title(url):
    title = ''
    try:
        res = requests.get(url, headers=HEADERS, timeout=1)
        # print type(res.content.decode(res.encoding))
        try:
            title = pq(res.content)('title').text().encode('gbk')
        except Exception, e:
            title = pq(res.content.decode(res.encoding))('title').text().encode('gbk')

    except Exception, e:
        print e
    return title

def get_url_list(query, site):
    url_list = []
    url = ''
    if site == 'baidu':
        url = 'http://www.baidu.com/s?ie=utf-8&f=8&rsv_bp=0&rsv_idx=1&tn=baidu&wd='
        try:
            res = requests.get(url + query, headers=HEADERS, timeout=1)
            result = pq(res.content).find('div').filter('.c-container')
            for i in range(len(result)):
                url_list.append(result.eq(i)('a').attr.href)
        except Exception, e:
            print e
    elif site == 'sogou':
        pass
    return url_list

def load_query_set(input_file_list):
    query_set = set()
    for input_file in input_file_list:
        with open(input_file) as f:
            for line in f:
                query_set.add(line.rstrip().split('\t', 1)[0].decode('gbk'))

    return query_set

def main():
    # title = get_title('http://www.baidu.com/link?url=aBq-n6YnjQSsTu3agpi-zvJQ4KO3SrxESs6j8In66RJWYAFkBBi5r299wC4pyP-liWSHbjxXV_V-C6wq1hiAhq')
    # print title
    # with open('data_baidu.txt', 'w') as f:
    #     f.write(title)
    url_list = get_url_list(u'索尼手机连不上网', 'baidu')
    for url in url_list:
        print url
   
    
    query_set = load_query_set([sys.argv[1]])
    # query_set_done = load_query_set(['./querys_done.txt'])
    
    # for query in query_set:
    #     if query not in query_set_done:
    #         query_left.add(query)
    # print len(query_left)
    
    with open(sys.argv[2], 'w') as fout:
        for query in query_left:
            url_list = get_url_list(query, 'baidu')
            for i in range(len(url_list)):
                if i > 6:
                    break
                title = get_title(url_list[i])
                if len(title) > 0 and '\n' not in title:
                    try:
                        line = query.encode('gbk') + '\t' + title + '\t' + str(i+1)
                    except Exception, e:
                        line = query + '\t' + title + '\t' + str(i+1)
                    # print line
                    fout.write(line + '\n')
                    time.sleep(1)

if __name__ == '__main__':
    main()
