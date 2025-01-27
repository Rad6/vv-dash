#!/bin/bash
DIR="$(dirname -- "$0")"

docker cp "${DIR}/Videos/." istream_server_nginx_container:/usr/local/nginx/html/
# Check if the source directory exists
# if [ -d "istream_server_nginx_container:/usr/local/nginx/html/merged30s" ]; then
#     # Copy the contents of the source directory to the parent directory
# docker cp istream_server_nginx_container:/usr/local/nginx/html/merged30s/* usr/local/nginx/html/
docker exec istream_server_nginx_container cp -r /usr/local/nginx/html/merged30s/. /usr/local/nginx/html/
#     echo "Contents of copied to"
# else
#     echo "Source directory does not exist."
# fi