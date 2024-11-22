<center><h1>linux</h1></center>



## 授权控制

### chown

- 更改所有者：`sudo chown newuser /path/to/directory`
- 更改组：`sudo chown :newgroup /path/to/directory`
- 递归更改：`sudo chown -R newuser:newgroup /path/to/directory`



### chmod

在使用命令 `ls -l` 后，对有每一个元素将会有 $10$ 个字符描述。

- 第一个字符：
  - `-`：表示这是一个普通文件。
  - `d`：表示这是一个目录。
  - `l`：表示这是一个符号链接。
  - 其他字符（例如 `b`、`c`）表示不同类型的特殊文件。
- 接下来的九个字符分为三个部分，每三个字符一组，每组代表不同用户的权限：
  - 用户 $User$ 权限：
    - `r`：用户读权限。
    - `w`：用户写权限。
    - `x`：用户执行权限。
  - 组 $Group$ 权限：
    - `r`：组读权限。
    - `-`：组无写权限。
    - `x`：组执行权限。
  - 其他用户 $Others$ 权限：
    - `r`：其他用户读权限。
    - `-`：其他用户无写权限。
    - `x`：其他用户执行权限。

对于上述后九个字符，每一个都可以进行单独改写：

```shell
chmod o-x example.txt
chmod g+r example.txt
chmod u=r example.txt
```

