#!/bin/sh
mkdir build
cd build
cmake ..
make
cd ..
sudo cp bin/SDFGen /usr/bin
