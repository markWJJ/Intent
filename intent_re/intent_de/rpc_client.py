#!/usr/bin/python3
# coding: utf-8
import json

import requests

res = requests.post('http://192.168.3.132:8086/intent', json={'data': ["今天天气怎么样", "尊享惠康等待期是多久"]})
res.json()

for e in res:
    e=e.decode()
    print(type(e))
    s_json=json.loads(e)
    for k in s_json:
        print(k)
