#!/usr/bin/python3
# coding: utf-8
import json

import requests

import requests
import time
url = "http://192.168.3.132:8084/intent"
# for index in range(1000):

start_time = time.time()



# datas = ['今天天气不错', '恶性肿瘤能否赔偿', '怎么申请理赔', '保险合同的受益人是谁？', '恶性葡萄胎赔不赔', '恶性肿瘤保不保', '重大疾病轻症疾病如何区别', '恶性肿瘤释义', '恶性肿瘤定义', '定义 恶性肿瘤', '身故保险金', '康爱保的身故保险金']
datas=['你好,请问有什么投保限制吗']
intent_ret = requests.post(url, json={'data': datas})
res = intent_ret.json()

print(res)

end_time=time.time()

print(end_time-start_time)