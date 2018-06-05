#!/usr/bin/python3
# coding: utf-8

from xmlrpc.server import SimpleXMLRPCServer
import tornado.ioloop
import tornado.web
import json
import sys
import logging
import objgraph
import argparse
from configparser import ConfigParser
from rule_new import intent_detection
# from intent_sys import Tree
# import os
import gc
PATH='.'

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')
_logger=logging.getLogger('intent')
Config=ConfigParser()
Config.read(PATH+'/Config.conf')
HOST=Config['intent']['host']
DEFAULT_PORT=int(Config['intent']['port'])
# TREE=Tree(PATH+'/data/意图识别.txt')
parser = argparse.ArgumentParser()

# 设置启动端口：
default_port = DEFAULT_PORT
parser.add_argument('-port', '--port', type=int, default=default_port, help='服务端口，默认: {}'.format(default_port))
args = parser.parse_args()

PORT = args.port

id=intent_detection()
def memory_usage_psutil():
    # return the memory usage in MB
    import psutil,os
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem

# @profile
def intent(sent_list):

    all_res=[]
    for index,sent in enumerate(sent_list):
        labels=id.intent_class(sent)
        _logger.info('is process:{}/{}/{}'.format(index,sent,labels))

        all_res.append(labels)
        del labels
        gc.collect()
    re_dict={}

    for sent,res in zip(sent_list,all_res):
        re_dict[sent]=res
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