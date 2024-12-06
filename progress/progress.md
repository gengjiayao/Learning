<center><h1>progress</h1></center>

### 2024.12.6

#### 进展

- 修改了一下 $udp$ 数据包



#### 问题

1. 修改哪个，`src` 和 `build` 文件夹之间是什么关系？（似乎已解决）

   - `/home/cuisine/ICT/rdma/conweave/ns-allinone-3.19/ns-3.19/build/ns3/seq-ts-header.h` 

   - `/home/cuisine/ICT/rdma/conweave/ns-allinone-3.19/ns-3.19/src/internet/model/seq-ts-header.h`

   - 好像知道为什么了

     - 我修改了 `build` 文件夹中的之后，编译一次，对应的权限就变成了只读了，所以我推测是简单的拷贝？

     - ```shell
       ......
       -rw-r--r--  1 cuisine cuisine 5.8K Nov 22 20:13 ctrl-headers.h
       -r--------  1 cuisine cuisine 3.9K Dec  6 11:06 custom-header.h
       -rw-r--r--  1 cuisine cuisine 2.9K Nov 22 20:13 data-calculator.h
       ......
       ......
       -rw-r--r--  1 cuisine cuisine 2.4K Nov 22 20:13 send-params.h
       -r--------  1 cuisine cuisine 2.1K Dec  6 11:26 seq-ts-header.h
       -rw-r--r--  1 cuisine cuisine 9.7K Nov 22 20:13 sequence-number.h
       ......
       ```

2. 奇怪的事情（已解决）

   - 使用 `Set` 函数不行，但是直接赋值可以？
     - 已解决：必须是 `this->test_udp = test_udp;` 不可以是 `test_udp = test_udp;`
     - 添加`this` 的对应的结果是 `578783551`，不添加 `this` 对应的结果为 `822695589`。

3. `IntHeader` 似乎并不是 `hpcc` 的专属，`timely` 也有 `INT` 头吗？

   - `/home/cuisine/ICT/rdma/conweave/ns-allinone-3.19/ns-3.19/build/ns3/int-header.h`

   - `/home/cuisine/ICT/rdma/conweave/ns-allinone-3.19/ns-3.19/src/network/utils/int-header.h`

   - ```cpp
         if (cc_mode == 7)  // timely, use ts
             IntHeader::mode = 1;
         else if (cc_mode == 3)  // hpcc, use int
             IntHeader::mode = 0;
         else  // others, no extra header
             IntHeader::mode = 5;
     ```

4. 用 `--cc hpcc` 就显示不出来我的测试内容，对应`744463944` 的结果，使用 `dcqcn` 和 `timely` 就没问题。

   - 在 `rdma-hw` 中检测，发现是没设置上，对应 `81927397`
   - 在 `RdmaHw::GetNxtPacket` 中检测，发现是设置上了，对应 `778110919`
   - 在 `qbb-netdevive.cc` 中检测，发现，前一部分正确，后一部分逐渐开始乱了，对应 `479157338`

5. 留行





### 2024.11.29

#### 进展

- 梳理了 $ns3$ 的启动流程，以及从发送方到接收方，再从接收方返回的函数调用过程。

  - 从 `run.py` 开始，执行 $waf$ 程序，并传入一个参数，这个参数是 $ns3$ 执行所需要的各种配置信息，包括 $cc\;mode$ 、$lb \;mode$ 、以及要发送的数据流的信息（数据长度，优先级，源、目的信息）

  - 从 `RdmaClient` 开始，沿着下面的路径进行发送，目前实现的发送是连续发送，不间断。

    ```
    RdmaClient --> RdmaDriver --> RdmaHw --> QbbNetDevice --> Swich节点发送
    ```

  - 发送完成后，沿着上方的路径，反着走，直到接收。

#### 问题

- `run.py` 里面并没有处理除了 $dcqcn$ 的情况，需要改 `run.py`。
- 修改的话应该怎么改？



### 2024.11.22

### 几大部分

- 发送方，发送数据包要按照一定的格式请求，写特定格式的请求包制作函数。
- 接收方，接收到请求后回复 $ACK$，写对应的 $ACK$ 包制作函数。
- 接收方，根据所有的请求组织出一个定制化的排序队列。
- 接收方，取出队列头的 $QP$ 请求，放入对应的 $SQ$ 中。
- 发送方，接收到 $ACK$ 做出下一个部分数据包的发送。



#### 进展

- $HPCC$ 的基本原理。
- $RDMA$ 相关学习。
- 看了 $HPCC$ 相关接收到 $ACK$ 后的处理。 



#### 相关问题

- $qLen$ 是干什么的？
- $INT$ 中只有 $5$ 个 $Hop$ 吗？
- $L1110$ 的计算不是很懂 
