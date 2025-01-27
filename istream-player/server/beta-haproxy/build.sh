#!/bin/bash

export DEBIAN_FRONTEND=noninteractive

apt update
apt install -y git ca-certificates gcc libc6-dev liblua5.3-dev libpcre3-dev libssl-dev libsystemd-dev make wget zlib1g-dev socat


# Install OpenSSL-quic
cd ~
git clone https://github.com/quictls/openssl
cd openssl
mkdir -p /opt/quictls/ssl
./Configure --libdir=lib --prefix=/opt/quictls
make
make install
echo /opt/quictls/lib | sudo tee -a /etc/ld.so.conf
ldconfig

# Install HAProxy
cd ~
git clone https://github.com/haproxy/haproxy.git
cd haproxy
make -j $(nproc) \
  TARGET=linux-glibc \
  USE_LUA=1 \
  USE_OPENSSL=1 \
  USE_PCRE=1 \
  USE_ZLIB=1 \
  USE_SYSTEMD=1 \
  USE_PROMEX=1 \
  USE_QUIC=1 \
  SSL_INC=/opt/quictls/include \
  SSL_LIB=/opt/quictls/lib \
  LDFLAGS="-Wl,-rpath,/opt/quictls/lib"

make install-bin