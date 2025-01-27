#!/bin/bash

set -e 

export DOLLAR='$'
envsubst < /etc/nginx/common/nginx.template.conf > /etc/nginx/nginx.conf
cat /etc/nginx/nginx.conf

function tc_stats() {
  while true
  do
      sleep 0.1
      echo "#EVENT" "TC_STAT" $(date +%s%3N) \
          $(tc -s -d class show dev eth0 | tr '\n' ' ') \
          $(tc -s -d qdisc show dev eth0 | tr '\n' ' ')
  done >> /run/event_logs_tc.txt
}

function tcpdump_collect() {
  tcpdump -i eth0 -s 65535 -w - > /run/server_out.pcap &
  tcpdump -i eth0 -s 65535 -w - > /run/server_in.pcap &
}

iptables -A OUTPUT -o eth0 -j NFLOG --nflog-group 1
iptables -A INPUT -i eth0 -j NFLOG --nflog-group 2

tc_stats &
tcpdump_collect &

nginx -g "daemon off;"
