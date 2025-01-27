#! /bin/bash

docker compose up -d
sh ./ha-proxy/apply-bw.sh
docker compose logs -f
