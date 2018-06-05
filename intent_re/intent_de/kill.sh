#!/bin/bash

path=`pwd`
echo ${path}

for i in `ps uax|grep 'python3'|grep rpc_server| grep ${path} | awk '{print $2}'` ; do
    kill -9 $i
done

echo "程序关闭"
