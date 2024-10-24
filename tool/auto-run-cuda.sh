

#!/bin/bash

# debs location here         
xdebs_dir=cuda-repo-cross-aarch64-ubuntu2004-11-4-local

pwdp=`pwd`

# to be save here 
xfiles=${pwdp}/${xdebs_dir}/xfiles 

echo "dpkg -x $deb begin ..."

# enter debs 
cd ${xdebs_dir} 

ls -l *.deb | awk '{print $9}' > tlog 
wc -l tlog

echo "=====read all deb files====="
for line in `cat tlog`
do
  echo $line'|======='
  dpkg -x $line ${xfiles}
done

echo "done"

cat << EOF > /dev/null
echo " !!! not reach here ... !!! "

export PATH=$PATH:/mnt/d/workspace/cuda-repo-ubuntu2004-11-4-local_11.4.20-470.161.03-1_amd64/xfiles/usr/local/cuda-11.4/bin 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/d/workspace/cuda-repo-ubuntu2004-11-4-local_11.4.20-470.161.03-1_amd64/xfiles/usr/local/cuda-11.4/targets/x86_64-linux/lib 

export PATH=$PATH:/mnt/d/workspace/nv-tensorrt-repo-ubuntu2004-cuda11.4-trt8.5.10.4-x86-host-ga-20221229_1-1_amd64/xfiles/usr/src/tensorrt/bin 

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/d/workspace/nv-tensorrt-repo-ubuntu2004-cuda11.4-trt8.5.10.4-x86-host-ga-20221229_1-1_amd64/xfiles/usr/lib/x86_64-linux-gnu 

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/d/workspace/cudnn-local-repo-ubuntu2004-8.6.0.174/xfiles/usr/lib/x86_64-linux-gnu 
EOF
