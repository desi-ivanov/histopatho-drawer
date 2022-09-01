#!/bin/bash
set -eou pipefail

# Build the docker image
docker build -t histopatho-drawer-server . --platform linux/amd64
