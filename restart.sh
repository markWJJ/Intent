#!/bin/bash
pip install xlrd
path=`pwd`
echo ${path}

for i in `ps uax|grep 'python'| awk '{print $2}'` ; do
    kill -9 $i
done

./start_server.sh

