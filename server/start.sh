#!/bin/bash
set -eou pipefail
FLASK_APP=main.py python -m flask run --host=0.0.0.0 --port=8080