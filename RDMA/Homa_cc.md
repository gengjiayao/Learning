 

<center><h1>Homa算法开发记录</h1></center>



## 1  发送方逻辑

按照窗口大小发送携带标记的 `udp` 数据包，



### 1.1  窗口相关

- 在 `main` 函数中有相应窗口的设置：
  - `clientHelper` 对象初始化的时候，有设定 `win`，此时值为 `maxbdp`，最大带宽时延积。
