#!/bin/bash

set -e

g++ -g -pthread -o build/test \
    src/main.cpp \
    -lglfw -lGLU -lGL -lXrandr -lXxf86vm -lXi -lXinerama -lX11 -lrt -ldl -lavcodec -lavformat -lswscale -lavutil

export FR=120
./build/test