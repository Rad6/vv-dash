#!/bin/bash
arguments=$1

DIR="$(dirname -- "$0")"
serverDockerPort=$(jq -r '.serverContainerPort' <<<${arguments})
serverDockerCpus=$(jq -r '.serverContainerCpus' <<<${arguments})
serverDockerMemory=$(jq -r '.serverContainerMemory' <<<${arguments})

dockerCupConfig=""
dockerMemoryConfig=""

if [ ${serverDockerCpus} != 0 ]; then
    dockerCupConfig="--cpus=${serverDockerCpus}"
fi

if [ ${serverDockerMemory} != 0 ]; then
    dockerMemoryConfig="--memory=${serverDockerMemory}g"
fi

docker ps -a -q --filter "name=istream_server_nginx_container" | grep -q . &&
    echo "Remove previous server docker container" && docker stop istream_server_nginx_container && docker rm -fv istream_server_nginx_container

docker run --name istream_server_nginx_container ${dockerCupConfig} ${dockerMemoryConfig} -p ${serverDockerPort}:80 -d istream_server_nginx_image

sh "${DIR}/Run/fetcher.sh"
if [ -f "${DIR}/Run/config.sh" ]; then
    docker cp "${DIR}/Run/config.sh" istream_server_nginx_container:/usr/local/nginx/html/
    docker exec istream_server_nginx_container bash /usr/local/nginx/html/config.sh
fi
    