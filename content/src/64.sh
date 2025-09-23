#!/bin/bash
for ((i=32;i<64;i++)); do
    echo "               ((i+($i*blockDim.x) < n) ? d_ivec[i+($i*blockDim.x)] : 0.0) +"
done
