#!/bin/bash

path=`pwd`
echo ${path}

cd ${path}

if [ `ps -aux|grep python3|grep rpc_server.py |grep ${path} | wc -l ` -le 0 ] ; then
    nohup python3 ${path}/rpc_server.py > /dev/null 2>&1 &
fi
    echo `date`, '程序, 启动'

/bin/bash

# yanghq@gpu2:~/intent_detection/intent_de$ docker run -it --restart=always --detach --name=intent_detection_18082 -p 18082:8082 -e MYDIR=/intent_detection --volume=$PWD:/intent_detection --workdir=/intent_detection -v /etc/localtime:/etc/localtime 899150993273.dkr.ecr.cn-north-1.amazonaws.com.cn/nlp:ubuntu-1221 /intent_detection/start.sh
# 837bca73e23e36577cb07eeb05770b3f8055d7477cdd510997e517416877ec28

