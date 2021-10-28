#!/bin/bash

if grep "127.0.0.1" /etc/hosts;  #这里即，如果在hosts文档中找到了127.0.0.1 这个字符串，那么

then
    echo "Everything OK"
else
    echo "Error, 127.0.0.1 is not in /etc/hosts"
fi # 'fi' closes  the if statement