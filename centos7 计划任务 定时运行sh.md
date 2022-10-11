1.安装 crontabs服务并设置开机自启

```shell
yum install crontabs （安装 crontabs）
systemctl enable crond （设为开机启动）
systemctl start crond（启动crond服务）
systemctl status crond （查看状态）
```



修改任务文件

```shell
vi /etc/crontab 
```



crontab文件内容如下：

```shell
SHELL=/bin/bash
PATH=/sbin:/bin:/usr/sbin:/usr/bin
MAILTO=root
 
# For details see man 4 crontabs
 
# Example of job definition:
# .---------------- minute (0 - 59)
# |  .------------- hour (0 - 23)
# |  |  .---------- day of month (1 - 31)
# |  |  |  .------- month (1 - 12) OR jan,feb,mar,apr ...
# |  |  |  |  .---- day of week (0 - 6) (Sunday=0 or 7) OR sun,mon,tue,wed,thu,fri,sat
# |  |  |  |  |
# *  *  *  *  * user-name  command to be executed
 
#每60分种执行
 */60 * * * * root /root/java/test1.sh
#每天1点20分 执行
 20 1 * * * root /root/java/test2.sh
#文件最后一定要留一个空行，不然命名：crontab /etc/crontab会报错
```

参数说明：

```
分钟(0-59) 小时(0-23) 日(1-31) 月(11-12) 星期(0-6,0表示周日) 用户名 要执行的命令

*/30 * * * root /usr/local/mycommand.sh (每天，每30分钟执行一次 mycommand命令)

* 3 * * * root /usr/local/mycommand.sh (每天凌晨三点，执行命令脚本，PS:这里由于第一个的分钟没有设置，那么就会每天凌晨3点的每分钟都执行一次命令)

0 3 * * * root /usr/local/mycommand.sh (这样就是每天凌晨三点整执行一次命令脚本)

*/10 11-13 * * * root /usr/local/mycommand.sh (每天11点到13点之间，每10分钟执行一次命令脚本，这一种用法也很常用)

10-30 * * * * root /usr/local/mycommand.sh (每小时的10-30分钟，每分钟执行一次命令脚本，共执行20次)

10,30 * * * * * root /usr/local/mycommand.sh (每小时的10,30分钟，分别执行一次命令脚本，共执行2次）
```



 

**保存生效 加载任务,使之生效：**

```
crontab /etc/crontab
```

查看任务：

```
crontab -l
crontab -u 用户名 -l （列出用户的定时任务列表）
```

PS：特别注意，crond的任务计划， 有并不会调用用户设置的环境变量，它有自己的环境变量，当你用到一些命令时，比如mysqldump等需要环境变量的命令，手工执行脚本时是正常的，但用crond执行的时候就会不行，这时你要么写完整的绝对路径，要么将环境变量添加到 /etc/crontab 中。