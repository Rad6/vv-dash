#!/bin/bash

set -e

# Listen for SIGINT signal and kill process on receiving signal
trap "exit" SIGINT SIGTERM

# Listen for network change requests
nc -kluv 4444 | /bin/sh &
echo "Listening for network change requests"

export DOLLAR='$'
envsubst </etc/nginx/common/nginx.template.conf > /run/nginx.conf
cat /run/nginx.conf

function tc_stats() {
  while true; do
    sleep 0.1
    echo "#EVENT" "TC_STAT" $(date +%s%3N) \
      $(tc -s -d class show dev eth0 | tr '\n' ' ') \
      $(tc -s -d qdisc show dev eth0 | tr '\n' ' ')
  done >>/run/event_logs_tc.txt
}

function tcpdump_collect() {
  tcpdump -i eth0 -s 65535 -w - >/run/server_out.pcap &
  tcpdump -i eth0 -s 65535 -w - >/run/server_in.pcap &
}

# iptables -A OUTPUT -o eth0 -j NFLOG --nflog-group 1
# iptables -A INPUT -i eth0 -j NFLOG --nflog-group 2

tc_stats &
tcpdump_collect &

# Try to load from config.json
PROTOCOL=$(cat "/run/config.json" | jq -r '.mod_downloader')
LOG_LEVEL=$(cat "/run/config.json" | jq -r '.serverLogLevel')
K_MAXIMUM_WINDOW=$(cat "/run/config.json" | jq -r '.K_MAXIMUM_WINDOW')
# Try to load from args
PROTOCOL=${PROTOCOL:-$1}
LOG_LEVEL=${LOG_LEVEL:-$2}
K_MAXIMUM_WINDOW=${K_MAXIMUM_WINDOW:-$3}

echo "Detected LOG_LEVEL: $LOG_LEVEL"
echo "Detected PROTOCOL: $PROTOCOL"
echo "Detected K_MAXIMUM_WINDOW: $K_MAXIMUM_WINDOW"
mkdir -p /run/server_log

if [[ "$PROTOCOL" == "tcp" ]]; then

  nginx -c /run/nginx.conf -g "daemon off;"

elif [[ "$PROTOCOL" == "quic" ]]; then
  extra_args=()

  if [[ "$LOG_LEVEL" == "debug" ]]; then
    extra_args+=("--quic-log")
    extra_args+=("/run/server_log")
  fi
  
  export K_MAXIMUM_WINDOW

  python3 /src/aioquic/server.py -v \
    --port 443 \
    -c /opt/nginx/certs/localhost.crt \
    -k /opt/nginx/certs/localhost.key \
    --quic-log /run/server_log -v
    # "${extra_args[@]}"

else
  echo "Invalid protocol: $PROTOCOL"
fi
