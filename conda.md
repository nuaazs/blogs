### 1. 查看基本信息：
```shell
% conda版本 %
conda -V

% 基本信息 %
conda info

% conda当前环境安装的所有的包 % 
conda list

% 查看当前存在的虚拟环境 %
conda env list 或 conda info -e

% 更新conda %
conda update conda
```

### 2. conda添加pypi源

```text
% 添加源 %
conda config --add channels 源地址

% 删除源 %
conda config --remove channels 源地址

% 源中搜索时显示通道地址 %
conda config --set show_channel_urls yes

% 恢复默认源 %
conda config --remove-key channels
```

### 3. conda-环境

```text
% 创建环境 %
conda create -n env_name python=x.x.x

% 激活环境 %
activate env_name

% 在虚拟环境中安装其他包 %
conda install -n env_name [package]

% 关闭当前虚拟环境，返回默认环境 %
deactivate env_name

% 删除环境 %
conda remove -n env_name --all

% 删除环境中的某个包
conda remove --name env_name [package]

% false即取消"退出时自动激活conda的base环境" %
conda config --set auto_activate_base false
```

### 4. conda安装cuda和cudnn

```text
% 安装cuda %
conda install cudatoolkit=版本号 -c 地址

% 安装cudnn %
conda isntall cudnn=版本号 -c 地址
```



### 5. 重命名cnv

- conda 其实没有重命名指令，实现重命名是通过 clone 完成的，分两步：
  1. clone 一份 new name
  2. delete old name

```shell
conda create -n tf --clone rcnn
```

```shell
Source:      /anaconda3/envs/rcnn
Destination: /anaconda3/envs/tf
Packages: 37
Files: 8463
```

```shell
conda remove -n rcnn --all
```

```

conda info -e
# conda environments:
#
crawl                    /anaconda3/envs/crawl
flask                    /anaconda3/envs/flask
tf                       /anaconda3/envs/tf
root                  *  /anaconda3

```