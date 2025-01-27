#!/bin/bash
BASE_DIR=$(realpath "$(dirname ${BASH_SOURCE})/../..")
echo $BASE_DIR
cd $BASE_DIR

docker run -d \
    --name http_file_server \
    -v $BASE_DIR/beta-emulator-quic/dataset:/var/www:ro \
    -p 8080:8080 \
    trinitronx/python-simplehttpserver
    
exit
docker container run -it --rm \
    --name haproxy_test \
    -v $BASE_DIR/beta-emulator-quic/beta-haproxy/haproxy.cfg:/usr/local/etc/haproxy/haproxy.cfg:ro \
    -v $BASE_DIR/certs:/etc/haproxy/certs/ \
    --sysctl net.ipv4.ip_unprivileged_port_start=0 \
    -p 443:443 \
    research_haproxy