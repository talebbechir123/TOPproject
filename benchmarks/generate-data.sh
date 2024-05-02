#!/bin/bash

generate_config_file() {
    size=$1
    cat << EOF
dim_x = $size
dim_y = $size
dim_z = $size
niter = $niter
EOF
}

niter=10

for ((size=100; size<=1000; size+=100))
do
    
        generate_config_file $size > "config_${size}_${i}.txt"
        echo "Config file config_${size}_${i}.txt generated"
   
done
