sudo apt-get update
sudo apt-get install nfs-kernel-server
sudo mkdir /var/nfs
sudo chown nobody:nogroup /var/nfs
sudo echo "/var/nfs $master_ip (rw,sync,no_subtree_check,no_,no_root_squash)" | tee -a /etc/exports
sudo exportfs -a
sudo service nfs-kernel-server start 
