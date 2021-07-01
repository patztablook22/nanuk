#!/usr/bin/env bash

cd $(dirname $0)
cmake -S src -B build
cd build
make
