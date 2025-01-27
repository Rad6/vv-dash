#!/bin/bash


docker compose exec ha-proxy chmod 777 ./bw-profile.sh
docker compose exec ha-proxy sh ./bw-profile.sh
