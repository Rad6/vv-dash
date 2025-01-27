#!/bin/sh
set -e

# set initial limited bw so that container won't have unlimited bw upon startup
INITIAL_BW=$(jq '.bandwidths[0]' bw-config.json)
echo "setting initial bw to ${INITIAL_BW}kbit"
tc qdisc add dev eth0 root netem rate ${INITIAL_BW}kbit

# first arg is `-f` or `--some-option`
if [ "${1#-}" != "$1" ]; then
	set -- haproxy "$@"
fi

if [ "$1" = 'haproxy' ]; then
	shift # "haproxy"
	# if the user wants "haproxy", let's add a couple useful flags
	#   -W  -- "master-worker mode" (similar to the old "haproxy-systemd-wrapper"; allows for reload via "SIGUSR2")
	#   -db -- disables background mode
	set -- haproxy -W -db "$@"
fi

exec "$@"
