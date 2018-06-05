#!/usr/bin/python3
# coding: utf-8

import tornado.ioloop
import tornado.web
import json
import sys
import logging
import argparse
import os,sys


from configparser import ConfigParser

from pipeline import Pipeline
import gc
PATH=os.path.split(os.path.realpath(__file__))[0]
sys.path.insert(0,PATH+'/intent_re/intent_de')
PATH='.'

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')
_logger=logging.getLogger('intent')
Config=ConfigParser()
Config.read(PATH+'/Config.conf')
HOST=Config['host']['host']
DEFAULT_PORT=int(Config['pipeline']['port'])
# TREE=Tree(PATH+'/data/意图识别.txt')
parser = argparse.ArgumentParser()

# 设置启动端口：
default_port = DEFAULT_PORT
parser.add_argument('-port', '--port', type=int, default=default_port, help='服务端口，默认: {}'.format(default_port))
args = parser.parse_args()

PORT = args.port

pipeline=Pipeline()
def memory_usage_psutil():
    # return the memory usage in MB
    import psutil,os
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem

# @profile
def intent(sent_list):

    re_dict={}
    res=pipeline.get_sent_result(sent_list)

    for sent,res in zip(sent_list,res):
        if isinstance(res,str):
            re_dict[sent]=[[res,1.0]]
        elif isinstance(res,list):
            ss=[]
            for ele in res:
                ss.append([ele,1.0])
            re_dict[sent]=ss
    del res
    gc.collect()
    return re_dict



class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("请使用post方法")

    def post(self):
        try:
            # if 'application/json' in content_type.lower():
            body_data = self.request.body
            if isinstance(body_data, bytes):
                body_data = self.request.body.decode('utf8')

            args_data = json.loads(body_data)
            data = args_data.get('data', [])
            ret = intent(data)
            result_str = json.dumps(ret, ensure_ascii=False)

            self.write(result_str)
        except Exception as e:
            self.write('{}')
            self.write("\n")
        sys.stdout.flush()

def make_app():
    return tornado.web.Application([
        (r"/intent", MainHandler),
    ])

def main():
    app = make_app()
    app.listen(PORT)
    tornado.ioloop.IOLoop.current().start()

if __name__ == "__main__":
    main()

# svr=SimpleXMLRPCServer((HOST, PORT), allow_none=True)
# svr.register_function(intent)
# svr.serve_forever()

# if __name__ == '__main__':
#
#     ss=[e.replace('\n','') for e in open('./FAQ_1.txt','r').readlines()]
#     sss=ss*20
#     intent(sent_list=sss)