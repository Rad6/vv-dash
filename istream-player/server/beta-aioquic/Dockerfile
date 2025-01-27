FROM nginx:1.23.3

RUN apt-get update && \
    apt install -y iproute2 procps tcpdump iptables

RUN apt install -y python3 python3-pip libssl-dev python3-dev build-essential libffi-dev

COPY aioquic /src/aioquic
RUN cd /src/aioquic && \
    pip3 install --upgrade pip && \
    pip3 install -r requirements.txt && \
    pip3 install -e .

RUN apt install -y jq socat dnsutils netcat sudo

RUN bash -c "setcap cap_net_admin+iep $(which tc)"

COPY certs               /opt/nginx/certs
COPY nginx.template.conf    /etc/nginx/common/nginx.template.conf
COPY container-start.sh     /etc/nginx/common/container-start.sh

RUN chmod -R 777 /var/cache/nginx /var/run /var/log/nginx

CMD ["bash", "/etc/nginx/common/container-start.sh"]
