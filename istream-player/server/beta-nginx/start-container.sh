#!/bin/bash

set -e 

pushd $(dirname "${BASH_SOURCE[0]}")/../..
export BASE_DIR=$(pwd)
popd

# Generate certificate
# openssl req -x509 -out localhost.crt -keyout localhost.key \
#     -newkey rsa:2048 -nodes -sha256 \
#     -subj '/CN=localhost' -extensions EXT -config <( \
#     printf "[dn]\nCN=localhost\n[req]\ndistinguished_name = dn\n[EXT]\nsubjectAltName=DNS:localhost\nkeyUsage=digitalSignature\nextendedKeyUsage=serverAuth")


# NGINX
# docker container run --name research-1 \
#     -v ${BASE_DIR}/beta-emulator-quic/dataset:/usr/share/nginx/html:ro \
#     -v ${BASE_DIR}:/research:ro \
#     -p 55000:80 \
#     --cap-add=NET_ADMIN \
#     -P -d research:latest

# NGINX-QUIC
docker run --name nginx-quic \
    -d -p 55000:80 -p 443:443/tcp -p 443:443/udp  \
    -v ${BASE_DIR}/certs:/opt/nginx/certs/ \
    -v ${BASE_DIR}/beta-emulator-quic/beta-ext/nginx.conf:/etc/nginx/nginx.conf \
    -v ${BASE_DIR}:/research:ro \
    -v ${BASE_DIR}/beta-emulator-quic/dataset:/etc/nginx/html:ro \
    --cap-add=NET_ADMIN \
    ymuski/nginx-quic