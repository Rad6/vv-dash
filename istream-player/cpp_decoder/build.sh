#!/bin/bash

# c++ -O3 -Wall -shared -std=c++11 -fPIC -I/usr/include/python3.10 istream_renderer.cpp -o build/istream_renderer$(python3-config --extension-suffix)

# gcc -g -pthread -o build/test src/*.cpp -lglfw -lGLU -lGL -lXrandr -lXxf86vm -lXi -lXinerama -lX11 -lrt -ldl -lavcodec -lavformat -lswscale -lavutil

# apt-get install libglfw3-dev libavformat-dev libswscale-dev
g++ -w -pthread -shared -std=c++11 -fPIC -static-libstdc++ \
    src/istream_module.cpp \
    -I/usr/include/python3.10 \
    -lglfw -lGLU -lGL -lXrandr -lXxf86vm -lXi -lXinerama -lX11 -lrt -ldl -lavcodec -lavformat -lswscale -lavutil \
    -o istream_renderer$(python3-config --extension-suffix)
