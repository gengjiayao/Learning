<center><h1>linux</h1></center>



## 查找动态库位置

- `ldd` 命令
  - `-u` 查找用不到的动态库。
- 一篇好文章：[查看linux下程序或者动态库到底依赖哪些so动态库以及对应的版本](https://blog.csdn.net/jinking01/article/details/120997165)





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

- 也可以直接使用类似 $775$、$644$ 这样的数字，直接赋值全部的 $9$ 种权限。
  - 对于一般的头文件，权限为 $644$。
  - 存储该头文件的文件夹，权限为 $755$。

