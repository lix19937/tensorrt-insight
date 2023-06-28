```
通用Linux 操作
NFS 挂载
服务端配置：

安装nfs：`sudo apt-get install nfs-kernel-server`
host PC创建共享文件夹用于mount，假设路径 /dir 注意 不要将源码放在共享目录
配置nfs
`sudo vim /etc/exports`
最后一行添加：/dir *(rw,sync,no_root_squash,no_subtree_check)
重启rpcbind  `sudo /etc/init.d/rpcbind restart`
重启nfs `sudo /etc/init.d/nfs-kernel-server restart`
客户端配置：

安装nfs `sudo apt-get update && sudo apt-get install nfs-common`
挂载目录： `sudo mount -t nfs <hostIP>:/dir /home/nvidia/nfs` ( /home/nvidia/nfs 是客户端文件夹)
optional： 设置开机启动自动挂载：`sudo touch /etc/profile.d/automount.sh && sudo echo "sudo mount -t nfs <hostIP>:/dir /home/nvidia/nfs" > /etc/profile.d/automount.sh`
取消挂载：

`sudo umount /home/nvidia/nfs`
SSHFS 挂载
客户端配置

安装`sshfs sudo apt-get install sshfs`
挂载目录 `sudo sshfs -p <Port> <HostUsername>@<HostIP>:/host/dir /local/dir`
服务端配置

远程工作站22端口屏蔽，开启8001端口
`sudo vi /etc/ssh/sshd_config`
解注Port 22，添加Port 8001
ssh免密登录配置

客户端 ssh-keygen，路径及密码可全默认回车。
`ssh-copy-id <HostUsername>@<HostIP> -p <Port>`
设置core dump生成路径
检查当前生成路径`cat /proc/sys/kernel/core_pattern`
开启core保存 `sudo ulimit -c unlimited`
检查是否开启core dump成功`ulimit –c`
永久设置新的生成路径`/sbin/sysctl -w kernel.core_pattern=/var/log/core_%e_%p_%t`
%% 单个%字符
%p 所dump进程的进程ID
%u 所dump进程的实际用户ID
%g 所dump进程的实际组ID
%s 导致本次core dump的信号
%t core dump的时间 (由1970年1月1日计起的秒数)
%h 主机名
%e 程序文件名
重启主机
检查当前生成路劲`cat /proc/sys/kernel/core_pattern`


ubuntu添加新的用户
添加新的用户：sudo adduser <username>
根据提示输入信息
添加权限：sudo vim /etc/sudoers

<username> ALL=(ALL:ALL) ALL
添加用户组：sudo usermod -aG 用户组 用户。用户添加后，运行一些可执行文件可能出错，需要将新增用户添加到相应的group里
查看用户组：cat /etc/group
查看所有用户：cat /etc/shadow
ubuntu2004安装搜狗输入法


远程工作站使用
代理proxy配置


域控制器MPD使用
MPD系统刷写方法
DEVKIT系统刷写方法
1. Clean previous installation:
    sudo -E apt-get -y --purge remove nv-driveos*
    sudo apt-get -y autoremove
    sudo rm -rf $NV_WORKSPACE
2. Installing host components on p3710
    sudo -i
    export NV_WORKSPACE=`your_workspace`
    dpkg -i nv-driveos-repo-sdk-linux-6.0.1.2-release-0002-29409022_6.0.1.2_amd64.deb
    dpkg -i /var/nv-driveos-repo-sdk-linux-6.0.1.2-release-0002-29409022/nv-driveos-foundation*
    dpkg -i /var/nv-driveos-repo-sdk-linux-6.0.1.2-release-0002-29409022/nv-driveos-linux*
    dpkg -i nv-driveos-foundation-p3710-specific-6.0.1.2-29409022_6.0.1.2-29409022_amd64.deb
3. Flash DRIVE OS Linux
    cd $NV_WORKSPACE/drive-foundation
    ./make/bind_partitions -b p3710-10-a01 linux
  Connect to the MCU console:
    minicom -w -D /dev/ttyACM1
  From the MCU console:
    tegrarecovery x1 on
    tegrareset x1
  Flash:
    tools/flashtools/bootburn/bootburn.py -b p3710-10-a01 -B qspi
From the MCU console:
    tegrarecovery x1 off
    tegrareset x1

4.copy ./tensorrt_path to orin devkit
    cd ./tensorrt_path
    sudo dpkg -i *.deb

CUDA/CUDNN/TENSORRT安装方式（MPD）
方法一：

cuda

sudo dpkg -i ./cuda-repo-ubuntu2004-11-4-local_11.4.15-1_arm64.deb
sudo apt-key add /var/cuda-repo-ubuntu2004-11-4-local/7fa2af80.pub
sudo apt update
sudo apt -y install cuda-toolkit-11-4
cudnn

sudo apt install ./cudnn-prune-87-repo-ubuntu2004-8-3-local_1.0-1_arm64.deb
sudo apt update
sudo apt install libcudnn8
sudo apt install libcudnn8-dev
sudo apt install libcudnn8-samples (该步骤安装不成功，转至方法二安装)
tensorrt

sudo dpkg -i nv-tensorrt-repo-ubuntu2004-cuda11.4-trt8.4.10.4-d6l-target-ga-20220413_1-1_arm64.deb    
sudo apt-key add /var/nv-tensorrt-repo-ubuntu2004-cuda11.4-trt8.4.10.4-d6l-target-ga-20220413/7fa2af80.pub
sudo apt update
sudo apt-get install tensorrt
(optional) sudo apt-get install tensorrt-safe (安装TensorRT safe 库)
方法二：

cuda

sudo dpkg -i ./cuda-repo-ubuntu2004-11-4-local_11.4.15-1_arm64.deb
cd /var/cuda-repo-ubuntu2004-11-4-local/
sudo dpkg -i *.deb
cudnn

sudo apt install ./cudnn-prune-87-repo-ubuntu2004-8-3-local_1.0-1_arm64.deb
cd /var/cudnn-prune-87-repo-ubuntu2004-8-3-local/
sudo dpkg -i *.deb
tensorrt

sudo dpkg -i nv-tensorrt-repo-ubuntu2004-cuda11.4-trt8.4.10.4-d6l-target-ga-20220413_1-1_arm64.deb  
cd /var/nv-tensorrt-repo-ubuntu2004-cuda11.4-trt8.4.10.4-d6l-target-ga-20220413/
sudo dpkg -i *.deb
CUDA/CUDNN/TENSORRT安装方式（X86_64）
方法一：

cuda

sudo dpkg -i ./cuda-repo-ubuntu2004-11-4-local_11.4.15-470.103.01-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu2004-11-4-local/7fa2af80.pub
sudo apt update
sudo apt -y install cuda-toolkit-11-4
cudnn

sudo apt install ./cudnn-local-repo-ubuntu2004-8.3.3.40_1.0-1_amd64.deb
sudo apt update
sudo apt install libcudnn8
sudo apt install libcudnn8-dev
sudo apt install libcudnn8-samples (该步骤安装不成功，转至方法二安装)
tensorrt

sudo dpkg -i nv-tensorrt-repo-ubuntu2004-cuda11.4-trt8.4.10.4-x86-host-ga-20220413_1-1_amd64.deb    
sudo apt-key add /var/nv-tensorrt-repo-ubuntu2004-cuda11.4-trt8.4.10.4-x86-host-ga-20220413/7fa2af80.pub
sudo apt update
sudo apt-get install tensorrt
(optional) sudo apt-get install tensorrt-safe (安装TensorRT safe 库)
方法二：

cuda

sudo dpkg -i ./cuda-repo-ubuntu2004-11-4-local_11.4.15-470.103.01-1_amd64.deb
cd /var/cuda-repo-ubuntu2004-11-4-local/
sudo dpkg -i *.deb
cudnn

sudo  dpkg -i  ./cudnn-local-repo-ubuntu2004-8.3.3.40_1.0-1_amd64.deb
cd /var/cudnn-local-repo-ubuntu2004*/
sudo dpkg -i *.deb
tensorrt

sudo dpkg -i ./nv-tensorrt-repo-ubuntu2004-cuda11.4-trt8.4.10.4-x86-host-ga-20220413_1-1_amd64.deb
cd /var/nv-tensorrt-repo-ubuntu2004-cuda11.4-trt8.4.10.4-x86-host-ga-20220413/
sudo dpkg -i *.deb
MPD 登录
使用ssh登录，可以配置config文件进行快速登录。配置方法

添加host条目到 `~/.ssh/config` ，没有文件的话需要新建

Host mpd30
HostName 10.94.xx.xx
Port 22
User mpd30
使用：`scp mpd30:/xx xx`  `ssh mpd30`
GPU主频修改
DOS6020使用了动态主频，在计算量不大时，GPU自动降频，节能。在DOS6030已经改回固定频率
修改方法，将GPU最低运行频率设置为最大运行频率：
检查最大运行频率：cat /sys/class/devfreq/17000000.ga10b/max_freq
将最低运行频率设置为最大频率：sudo vim /sys/class/devfreq/17000000.ga10b/min_freq 
注意：重启后需要重新设置

MAC地址修改
查看运行的GPU进程
sudo lsof | grep nvgpu
查看内存实时带宽
sudo emc_log -client readtally -client writetally -window 100 -length 10
查看CPU&GPU实时负载
tegrastats
nsys 工具安装
MPD 和 DEVKIT 刷机后，默认没有nsys工具。需要自行安装

下载对应版本的 NsightSystems 安装包    
提取 aarch64 平台可执行文件
'ar x NsightSystemsxxxxxx.deb'
'tar xpf data.tar.gz'
把解压的 aarch64 相关可执行文件拷贝到mpd：`opt/nvidia/nsight-systems/2023.1.3/target-linux-tegra-armv8`
板子上有类似路径，合并即可
profile文件可视化，可以使用`opt/nvidia/nsight-systems/2023.1.3/bin/nsys-ui'，运行在 x86_64 上
环境配置
docker 挂载GPU
将GPU挂载到container的命令参考 `docker run -itd --gpus all --name <容器名> -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility -e NVIDIA_VISIBLE_DEVICES=all <镜像名>`
Issue
docker run 报错： `docker: Error response from daemon: could not select device driver "" with capabilities: gpu.`

安装 apt-get install nvidia-container-toolkit （如果无法安装，往下看）

重启docker服务 systemctl restart docker

找不到 nvidia-container-toolkit ，无法安装 (如果无法下载，往下看)

$ curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey > gpgkey && sudo apt-key add gpgkey
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
$ curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list > nvidia-docker.list && cp nvidia-docker.list /etc/apt/sources.list.d
$ sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
$ sudo systemctl restart docker


$ cd <存放 gpgkey和nvidia-docker.list文件的文件夹>
$ sudo apt-key add gpgkey
$ cp nvidia-docker.list /etc/apt/sources.list.d
$ sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
$ sudo systemctl restart docker
问题记录：
APT 安装程序时报错：
dpkg: unrecoverable fatal error,

aborting: unknown system user '*****' in statoverride file;

the system user got removed before the override, which is most probably a packaging bug,
to recover you can remove the override manually with dpkg-statoverride

E: Sub-process /usr/bin/dpkg returned an error code (2)

解决方法：

sudo vi /var/lib/dpkg/statoverride

// 删除有关 “*****” 的行，或行里对应的文件。
```
