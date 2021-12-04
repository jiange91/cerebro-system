sudo apt-get update
sudo apt-get install -y nfs-common
sudo mkdir /var/nfs
sudo mount $master_ip:/var/nfs /var/nfs