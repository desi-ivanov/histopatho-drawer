#!/bin/bash
set -e
rest=$@
docker run -it --rm \
        -v "$(pwd)":/workspace \
        --workdir=/workspace \
        -p 8080:8080 \
        --platform linux/amd64 \
        "histopatho-drawer-server" $@

