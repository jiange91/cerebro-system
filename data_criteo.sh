#!/usr/bin/env bash

wget -q -O /tmp/gdrive.tar.gz "https://github.com/prasmussen/gdrive/releases/download/2.1.1/gdrive_2.1.1_linux_386.tar.gz"
mkdir /tmp/gdrive && tar -xf /tmp/gdrive.tar.gz -C /tmp/gdrive
chmod +x /tmp/gdrive/gdrive
/tmp/gdrive/gdrive download 1J2llJbioU9OdXZCNzZNNHpNNOf6lv8Qx --recursive --path /var/nfs/criteo
