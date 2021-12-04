#!/bin/bash
echo "start jupyter notebook on port: 1111"
jupyter notebook --no-browser —port 1111
echo "start tensorboard on port: 2222"
tensorboard --logdir=/var/nfs/test/runs/logs  --port=2222