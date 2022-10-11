Github无法push问题：

可能是由于切换到了女朋友的git账号，再切换回来之后，发现git push每次都提示要输入git@github.com的密码。

没有绑定到我的账号，所以导致无法push，设置ssh密钥也无效。

```shell
git push -u origin main
```



## 第一种解决方案

```shell
git push https://user:passward@github.com/<GITHUB_USERNAME>/<REPOSITORY_NAME>.git
```

还是错误，提示这种方法已经被取缔。



[solution](https://techglimpse.com/git-push-github-token-based-passwordless/)

所以需要通过github的token认证直接更新：

```shell
git push https://<GITHUB_ACCESS_TOKEN>@github.com/<GITHUB_USERNAME>/<REPOSITORY_NAME>.git
```

但是并没有治本，下次直接`git push -u origin main`还是同样的问题，暂时先这样处理把。



## tips1

shell设置proxy和取消(加速github)

```shell
alias proxy="export http_proxy=http://127.0.0.1:1090 https_proxy=http://127.0.0.1:1090"
alias unset_proxy="unset http_proxy https_proxy"
```

[参考博客](https://www.zhihu.com/question/40933654)



## tips2

不小心commit了过大的文件的解决方法。

解决方法，查看log，撤销commit，删除文件，重新add+commit再push。

撤销commit文件:`git reset xxxxxxxxxxxxx`

[参考博客](https://www.cnblogs.com/makalochen/p/14484820.html)

