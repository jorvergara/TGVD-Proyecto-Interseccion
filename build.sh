#!/bin/bash

chmod +x compile.sh
chmod +x run.sh

git submodule update --init --recursive
if [ -d build ]; then
  rm -r build/
fi
mkdir -p build/
cd build 
cmake ..
make
cd ..