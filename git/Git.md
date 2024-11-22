<center><h1>Git</h1></center>

### 一、科学上网相关的Git配置

```
# ~/.ssh/config
Host github.com
    HostName ssh.github.com
    User git
    Port 443
    IdentityFile ~/.ssh/id_ed25519
    ProxyCommand nc -X 5 -x 127.0.0.1:7890 %h %p
```

1. 默认的 $ssh$ 端口是 $22$，这里制定 $Port$ 为 $443$，因为 $ssh.github.com$ 的端口为 $443$。
2. 制定的认证文件要与 $github$ 上设置的一致
3. $ProxyCommand$ 命令表示一个代理命令
   - `nc` 表示通过 $Netcat$ 发送 $ssh$ 流量；
   - `5` 表示使用 $SOCKS5$ 代理；
   - `127.0.0.1:7890` 指代代理服务器地址和端口，通常是一个本地运行的代理；
   - `%h`和`%p`是占位符，分别代表主机名和端口名，即 $ssh.github.com$ 和 $443$ 。



### 二、
