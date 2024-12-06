<center><h1>Conweave Learning</h1></center>



### ns3启动流程

- `run.py`：`os.system` 函数执行 `./waf --run 'scratch/network-load-lanbace {config_name}`，`config_name` 为配置文件。
  - `config_name` 一律保存在了 `/mix/output/` 文件夹下的 `config.txt`。
- `network-load-balance.cc`：`main` 函数中 `conf` 读入上方 `config_name` 的参数。（ $Line\;737$ ）
  - `conf` 中有 `FLOW_FILE`、`CC_MODE` 等各种参数。



### 自上而下协议栈

#### 发送数据

``` 
RdmaClient --> RdmaDriver --> RdmaHw --> QbbNetDevice --> Swich节点发送
```

- `RdmaClientHelper` 获得由 `conf` 读入的各种信息，请求 $rdma$ 服务。
- `RdmaClient::StartApplication` 调用 `rdma-driver.cc` 中的 `RdmaDriver::AddQueuePair`。
- `m_rdma->AddQueuePair` 调用 `rdma-hw.cc` 中 `RdmaHw::AddQueuePair`，初始化 $qp$ 的一些信息。
- `RdmaHw::AddQueuePair` 调用 `NewQp(qp)` 真正通知网卡设备，创建 $qp$。
- `NewQp` 调用 `DequeueAndTransmit` 函数，这个函数是交换机函数或者主机函数（多用的）。
- `DequeueAndTransmit` 函数：
  - 函数内先调用 `DequeueQindex` 函数，内部继续调用 `m_rdmaGetNxtPkt` 函数，这是一个回调函数，用于生成数据包（千辛万苦终于找到你！）
  - 函数内再调用 `QbbNetDevice::TransmitStart` 函数实际发送。
-  `QbbNetDevice::TransmitStart` 函数：
  - 记录完成时间等具体信息，调用 `QbbNetDevice::TransmitComplete` 函数。
    - `TransmitComplete` 函数中会安排在未来一个时间点调用 `DequeueAndTransmit` 函数，这个时间点的安排就体现了发送的延迟，等到到达时间点，将调用`DequeueAndTransmit` 函数，继续获取下一个数据包进行发送。
  - 调用 `QbbChannel::TransmitStart` 函数，紧接着调用 `QbbNetDevice::Receive` 函数进入接收状态。



#### 接收数据

- `QbbChannel::TransmitStart` 中调用接收函数 `QbbNetDevice::Receive`。
- `QbbNetDevice::Receive` 中接收到数据包，判断传递到 $switch$ 还是 $nic$。
  - 传递到 $switch$ ：调用`m_node->SwitchReceiveFromDevice` 函数 。
    - 调用 `SwitchNode::SwitchReceiveFromDevice` 函数。
    - 调用 `SendToDev` 函数。
      - 根据 $lb\;mode$ 走 $Conga$ 或者 $ConWeave$。
      - 都不是就调用 $SendToDevContinue$ 函数。
    - 调用 `DoSwitchSend` 函数。
    - 调用 `QbbNetDevice::SwitchSend` 函数。
    - 调用 `DequeueAndTransmit` 函数，继续传递给下一跳。
  - 传递到 $nic$ ：调用 `m_rdmaReceiveCb` 回调函数，这个回调函数对应 `RdmaHw::Receive` 函数。
- 接收到的比如是 `UDP` 包，调用 $ReceiveUdp$ 函数处理。
- $ReceiveUdp$ 函数中会产生 $ACK$ 或 $NACK$。
- 调用 `RdmaEnqueueHighPrioQ`  函数将生成的 $ACK$ 包进入 $ACK$ 队列，这个包提前设置成了高优先级，以后在 `DequeueAndTransmit` 函数中、`DequeueQindex` 函数中通过高优先级先去发 $ACK$。
- 调用 `TriggerTransmit` 函数，发送 $ACK$ 包，发送过程见上方“发送部分”。



### 自定义数据包字段

#### 修改 $udp$ 报文头

- `seq-ts-header` 相关修改

  - 两处？`seq-ts-header.h` 的 `public` 字段添加 `test_udp` 字段。
  - `seq-ts-header.cc` 中 `SeqTsHeader::GetHeaderSize`函数：
    - 也就是`seq-ts-header.h` 中 `virtual uint32_t GetSerializedSize (void) const;` 的具体实现， `+4`。
  - `seq-ts-header.cc` 中 `SeqTsHeader::Serialize` 函数：
    - 添加 `i.WriteHtonU32(test_udp);` 
  - `seq-ts-header.cc` 中 `SeqTsHeader::Deserialize` 函数：
    - 添加 `test_udp = i.ReadNtohU32();`

  

- `custom-header` 相关修改

  - 两处？`custom-header.h` 的 `udp` 结构体中添加 `test_udp` 字段。
  - `custom-header.cc` 中 `CustomHeader::GetUdpHeaderSize` 函数：
    - `+ sizeof(udp.test_udp)`
  - `custom-header.cc` 中 `CustomHeader::Serialize` 函数：
    - 添加 `i.WriteHtonU32(udp.test_udp);`
  - `custom-header.cc` 中 `CustomHeader::Deserialize` 函数：
    -  `UDP + SeqTsHeader` 中添加`udp.test_udp = i.ReadNtohU32();`



#### 修改 $udp$ 相关处理函数

- `seq-ts-header` 相关修改
  - `seq-ts-header.h` 头文件
    - 两处？添加 `SetTestUDP` 和 `GetTestUDP` 的函数声明。
  - `seq-ts-header.cc` 实现文件
    - 添加 `SeqTsHeader::SetTestUDP` 和 `SeqTsHeader::GetTestUDP` 的函数实现。



#### $udp$ 相关成员变量值的设置

- `RdmaHw::GetNxtPacket` 相关修改：
  - 设置 `test_udp` 的值：`seqTs.SetTestUDP(20240000);` 



#### $udp$ 相关修改在接收方的检查

- 在 `generate ACK or NACK` 上方添加检查：

- ```cpp
  if (ch.udp.test_udp == 20240000) {
      printf("receive success test_udp: %d\n", ch.udp.test_udp);
  }
  ```

- 



### CustomHeader相关

总的来说，这个定制化的报文头是将传输中的数据包的报文头解析后保存下来的数据。

- 在接收数据时 `QbbNetDevice::Receive` 中，首先会去定义一个 `CustomHeader ch`。
- 调用 `packet->PeekHeader(ch)` 函数去填充 `ch`，调用 `header.Deserialize` 函数
  - `PeekHeader` 是 `packer.cc` 中的函数，传入的参数是 `CustomHeader` 数据类型的 `ch`，由于 `CustomHeader` 继承于 `Header`，所以 `Header` 中具体的虚函数由 `CustomHeader::Deserialize` 具体完成。

