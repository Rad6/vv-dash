FROM golang:latest AS builder

ENV DATAPLANE_MINOR 2.8.1
ENV DATAPLANE_URL https://github.com/haproxytech/dataplaneapi.git

RUN git clone "${DATAPLANE_URL}" "${GOPATH}/src/github.com/haproxytech/dataplaneapi"
RUN cd "${GOPATH}/src/github.com/haproxytech/dataplaneapi" && \
    git checkout "v${DATAPLANE_MINOR}" && \
    make build && cp build/dataplaneapi /dataplaneapi

FROM alpine:3.18


LABEL Name HAProxy
LABEL Release Community Edition
LABEL Vendor HAProxy
LABEL Version 2.8.3
LABEL RUN /usr/bin/docker -d IMAGE

ENV HAPROXY_BRANCH 2.8
ENV HAPROXY_MINOR 2.8.3
ENV HAPROXY_SHA256 9ecc6ffe67a977d1ed279107bbdab790d73ae2a626bc38eee23fa1f6786a759e
ENV HAPROXY_SRC_URL http://www.haproxy.org/download

ENV DATAPLANE_MINOR 2.8.1
ENV DATAPLANE_SHA256 ec685350576898bb0a00d656f427f5da7c28043551a3e0050c1b5301547e235a
ENV DATAPLANE_URL https://github.com/haproxytech/dataplaneapi/releases/download

ENV HAPROXY_UID haproxy
ENV HAPROXY_GID haproxy

COPY --from=builder /dataplaneapi /usr/local/bin/dataplaneapi

RUN apk add --no-cache ca-certificates jq && \
    apk add --no-cache --virtual build-deps gcc libc-dev \
    linux-headers lua5.4-dev make openssl openssl-dev pcre2-dev tar \
    zlib-dev curl shadow && \
    curl -sfSL "${HAPROXY_SRC_URL}/${HAPROXY_BRANCH}/src/haproxy-${HAPROXY_MINOR}.tar.gz" -o haproxy.tar.gz && \
    echo "$HAPROXY_SHA256 *haproxy.tar.gz" | sha256sum -c - && \
    groupadd "$HAPROXY_GID" && \
    useradd -g "$HAPROXY_GID" "$HAPROXY_UID" && \
    mkdir -p /tmp/haproxy && \
    tar -xzf haproxy.tar.gz -C /tmp/haproxy --strip-components=1 && \
    rm -f haproxy.tar.gz && \
    make -C /tmp/haproxy -j"$(nproc)" TARGET=linux-musl CPU=generic USE_PCRE2=1 USE_PCRE2_JIT=1 \
                            USE_TFO=1 USE_LINUX_TPROXY=1 USE_GETADDRINFO=1 \
                            USE_LUA=1 LUA_LIB=/usr/lib/lua5.4 LUA_INC=/usr/include/lua5.4 \
                            USE_PROMEX=1 USE_SLZ=1 \
                            USE_OPENSSL=1 USE_PTHREAD_EMULATION=1 \
                            all && \
    make -C /tmp/haproxy TARGET=linux2628 install-bin install-man && \
    ln -s /usr/local/sbin/haproxy /usr/sbin/haproxy && \
    mkdir -p /var/lib/haproxy && \
    chown "$HAPROXY_UID:$HAPROXY_GID" /var/lib/haproxy && \
    mkdir -p /usr/local/etc/haproxy && \
    ln -s /usr/local/etc/haproxy /etc/haproxy && \
    cp -R /tmp/haproxy/examples/errorfiles /usr/local/etc/haproxy/errors && \
    rm -rf /tmp/haproxy && \
    chmod +x /usr/local/bin/dataplaneapi && \
    ln -s /usr/local/bin/dataplaneapi /usr/bin/dataplaneapi && \
    touch /usr/local/etc/haproxy/dataplaneapi.yml && \
    chown "$HAPROXY_UID:$HAPROXY_GID" /usr/local/etc/haproxy/dataplaneapi.yml && \
    apk del build-deps && \
    apk add --no-cache openssl zlib lua5.4-libs pcre2 && \
    rm -f /var/cache/apk/*

# RUN apt-get -qq update && apt-get install -qq -y --no-install-recommends \
#     iperf \
#     iproute2 \
#     iptables \
#     iputils-ping \
#     python3.4 \
#     python3-pip

ENV PYTHONUNBUFFERED=1
RUN apk add --update --no-cache python3 && ln -sf python3 /usr/bin/python
RUN apk add iperf
RUN apk add iproute2
RUN apk add iptables
RUN apk add iputils-ping
RUN python3 -m ensurepip
RUN pip3 install --no-cache --upgrade pip setuptools

# RUN rm -rf /var/lib/apt/lists/* 

RUN pip3 install setuptools 
RUN pip3 install tcconfig

COPY haproxy.cfg /usr/local/etc/haproxy
# RUN ["chmod", "+x", "/docker-entrypoint.sh"]
COPY docker-entrypoint.sh /
COPY bw-profile.sh /
COPY bw-config.json /

STOPSIGNAL SIGUSR1

ENTRYPOINT ["sh", "/docker-entrypoint.sh"]
CMD ["haproxy", "-f", "/usr/local/etc/haproxy/haproxy.cfg"]