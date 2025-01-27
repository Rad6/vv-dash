#! /bin/bash

# Read bandwidths from JSON file
BANDWIDTHS=$(jq '.bandwidths[]' bw-config.json)
# Read duration for sleep from JSON file
DURATION=$(jq '.durations' bw-config.json)

# Loop through the bandwidth numbers
for i in $BANDWIDTHS
do
    tc qdisc del dev eth0 root
    echo "setting available bw to ${i}kbit for ${DURATION} seconds"
    tc qdisc add dev eth0 root netem rate ${i}kbit
    sleep $DURATION;
done

