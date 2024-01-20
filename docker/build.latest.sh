#!/bin/bash

unset KUBECONFIG

cd .. && docker build --platform=linux -f docker/Dockerfile.latest \
             -t yancyyu/chatgpt-on-wechat .

docker tag yancyyu/chatgpt-on-wechat yancyyu/chatgpt-on-wechat:$(date +%y%m%d)
