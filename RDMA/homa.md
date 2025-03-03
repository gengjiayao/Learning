<center><h1>Homa</h1></center>

## 整体运行逻辑

### 从应用程序接受消息开始

#### 第一次发送

- $handleMessage$：
  - 消息传入，如果是上层 $appIn$，则进入 `sxController.processSendMsgFromApp` 进行处理。
- $processSendMsgFromApp$：
  - 消息数据包括：请求数据、未调度数据。
  - 通过 `getReqUnschedDataPkts` 函数将消息的数据处理成许多数据包，存放在 `reqUnschedDataVec` 中。
  - 通过 `forward_as_tuple` 隐式调用了 `outboundMsg` 的构造函数，这使得自定义的消息中含有了各个数据包大小的信息。
  - 调用 ` prepareRequestAndUnsched` 函数准备请求与未调度的数据包。
- $prepareRequestAndUnsched$：
  - 通过 `getUnschedPktsPrio` 函数得到每一个数据包对应的优先级。
  - 在 `prioUnschedBytes` 的数组中，以两位为一组，保存全部数据包的优先级和未调度数据大小。
  - 不断构造数据包 `HomaPkt`，实际的变量名称为 `unschedPkt`，构造完成便将其放入消息对象的 `txPkts` 优先队列中。
  - `txPkts` 优先队列的排序逻辑如下：
    - 看起始字节大小，起始字节大的优先；
    - 看创建时间早晚，创建时间晚的优先；
    - 看优先级高低，优先级高的优先。
  - （这样看来，每个消息都有一个优先队列？这个排序真的可以这样吗？）
- 把自定义的消息插入到 `outbndMsgSet` 中，此时 `outbndMsgSet.size()` 是 $1$。
- $sendOrQueue$：
  - 此时调用传入的参数是空，也就是说 `msg` 为空，分支判断逻辑将进入下方传参为空的判断中。
  - 如果当前 `sendtimer` 被调度了，证明当前忙，直接返回，否则继续向下判断。
  - 调用 `getTransmitReadyPkt` 函数，从 `txPkts` 中获得一个最优先发送的包，给到 `sxPkt`。
  - 调用 `sendPktAndScheduleNext` 函数，处理 `sxPkt`。
- 如果当前消息分解的数据包还有剩余，也就是 `txPkts` 不为空时，会继续把这个消息插入到 `outbndMsgSet` 中。
  - `outbndMsgSet` 这个集合的排序逻辑如下（以 `SRBF` 为例）：
    - 消息剩余的少的优先；
    - 创建时间早的优先；
    - 消息 $id$ 小的优先。
- $sendPktAndScheduleNext$：
  - 获得实际传输的字节大小 `bytesSentOnWire`。
  - `nicLinkSpeed` 以 `Gbps` 为单位，所以需要乘 `1e-9`。 
  - 通过 `sxPktDuration` 更新下一次发送的时间。
  - 调用 `transport->socket.sendTo` 发送数据包，并且调用 `transport->scheduleAt` 规划下一次的 `sendTimer`，这里的 `sendTimer` 是一个自消息，会在 `handleMessage` 中被接受。
- `sendTo` 函数调用 `sendToUDP` 实际发送 `udp` 消息。
- 此后，一方面，`handleMessage` 函数等待接收 `udpIn`消息，也就是上面 `sendTo` 发送的消息；另一方面，`handleMessage` 函数等待 `scheduleAt` 调度的自消息 `sendTimer`。



#### 第一次接收

- $handleMessage$：
  - 这次接受的是 `udpIn` 消息，调用 `handleRecvdPkt` 函数处理。
- $handleRecvdPkt$：
  - 根据接收的数据包，更新了一些相关活动周期的时间信息（这一块儿没看懂）
  - 根据接收的数据包调用处理函数 `processReceivedPkt`。
- $processReceivedPkt$：
  - 带宽浪费的判定：
    - 设定两次接收到数据包的时间间隔为 $t_1$，接收一个数据包所需要的时间是 $t_2$，在 $t_1$ 时间内传输的授权数据包延时为 $t_3$。
    - 如果 $t_1 \gt t_2 + t_3$ 并且当前还有等待授权方，此时便存在了带宽浪费。
  - 计算带宽浪费估计值：
    - 





## 其他相关

- $Homa$ 是消息驱动的模拟器
  - $AppMessage$ 是应用层传递的消息，该消息通过处理，处理成 $OutboundMessage$。
  - $OutboundMessage$ 是 $homa$ 层面的消息，进行后续处理和调度。
- $InboundMessage$ 与 $OutboundMessage$
  - $InboundMessage$
    - 核心：处理从网络接收的数据，关注如何高效地接收和重组消息；管理接收端发送的授权，控制发送端的数据发送；主要处理接收数据包的重组和完整性检查。
    - $GrantList$：

  - $OutboundMessage$
    - 核心：处理向网络发送的数据，关注如何高效地准备和发送消息；根据接收端的授权来发送数据，管理调度和非调度数据包的发送；管理数据包的准备和发送，包括请求包、非调度数据包和调度数据包。
    - 




## 授权相关机制

- 执行过程

```cpp
handleMessage --> processSendMsgFromApp --> sendOrQueue --> sendPktAndScheduleNext --> socket.sendTo (并且 scheduleAt 下一次 sendTimer)
							|  (处理第一个未计划数据包)                     (发送并安排下一个数据包)       (实际发送)
							|
              --> handleRecvdPkt --> processReceivedPkt、handleInboundPkt --> handlePktArrivalEvent --> sendAndScheduleGrant --> sendOrQueue
                 (对应2b两种分支) |    (接收数据方-接收方)
  															 |
                                 --> processReceivedGrant -->  sendOrQueue -->
  																	  (接收授权方-发送方)
```



- $Unsched Send$ ：
  - $Unsched$ 发送完毕后的下一个 $SendTimer$ 到来：
    -  由于该消息在 $outbndMsgSet$ 中被删除，则直接在 $sendOrQueue$ 返回；如果有其他从应用层请求的 $Unsched$ 包发送，会发送出去，直到 $outbndMsgSet$ 没有消息了。
  - 下面是接收方接收到数据包后，进行一些必要的处理 $processReceivedPkt$。
    - 接收方 $handleRecvdPkt$，检查带宽是否浪费，如果浪费看看是否可以再早一点授权充分利用带宽（具体的算法没仔细看，属于"过度"授权的机制）
    - 构造/找到发送者的 $SenderState$，每一个发送者对应一个$grantTimer$。
    - $SchedSenders$ 是有排序的，具体的排序同下方排序队列逻辑，每次接收到一个新的数据包，可能 $SchedSenders$ 的排序是会变化的。
  - $handleInboundPkt$，处理收到的数据包，有五种情况，这里有添加。
  - $handlePktArrivalEvent$ -- $sendAndScheduleGrant$，处理发送 $grant$ 数据包，安排 $grantTimer$。



```cpp
GRANT
  processGrantTimers --> handleGrantTimerEvent --> handleGrantSentEvent --> sendAndScheduleGrant --> sendOrQueue --> sendPktAndScheduleNext(有了transport->sendTimer) --> 走send
                                                                                                (outGrantQueue.push(sxPkt))
```



* $handleMessage$

  ```cpp
  // handleMessage (HomaTransport.cc 237)
  (1) 判断是内部消息
      a. 根据具体消息类型进行特定处理
  (2) 判断是外部消息
    	a. 是来自应用层的消息，后续需要进行第一次 Unsched 数据包的准备（主要相关发送方的处理逻辑）
      b. 是来自网络层的数据包，可能是发送方接收到授权数据包、也可能是接收方收到的发送方发来的数据包（两种情况）
  ```

  

* $processSendMsgFromApp$ 第一个 $Unsched$

  ```cpp
  // processSendMsgFromApp (HomaTransport.cc 491)
  
  1. outboundMsg->prepareRequestAndUnsched() /** 根据消息，准备要发送的未调度的数据包 */
  2. auto insResult = outbndMsgSet.insert(outboundMsg); /** 存储指针到set数组中 用自定义sort排序 */
  
  
  
  (1) rxAddrMsgMap[destAddr].insert(outboundMsg); /** 哈希插入set 仅仅记录一下 */ 表明SendController可以对多个地址！一个应用程序一个？
  ```

  

* $processReceivedPkt$

  ```cpp
  // HomaTransport::ReceiveScheduler::processReceivedPkt(HomaPkt* rxPkt)
  
  // 这个函数用于处理到达接收方传输层的数据包
  ```

  

* $sendOrQueue$

  ```cpp
  // sendOrQueue (HomaTransport.cc 579)
  (1) 有三种情况
      a. 要发送到网络的授权数据包
      b. 发送定时器信号，表示应该发送下一个准备好的数据包
      c. 参数为NULL，此时表示一个数据包已经准备好缓存到了SendController队列中
  ```

  

* $handleRecvdPkt$

  ```cpp
  // sendOrQueue (HomaTransport.cc 579)
  ```

  

* $sendAndScheduleGrant$

  ```cpp
  // sendAndScheduleGrant (HomaTransport.cc 1816)
  // 调度授权的流程
  (1) 根据给定的授权优先级，决定是否取消已经准备好的授权定时器
  (2) 如果优先级更高，给当前高优先级的授权包做发送，并对当前的数据包做下一次授权定时器的准备
  ```

  

* 排序队列逻辑：

  ```cpp
  // class CompareBytesToGrant (HomaTransport.h)
  (1) 首先比较需要授权的剩余字节数量（剩余少的优先）
  (2) 再次比较整体消息的大小（小的优先）
  (3) 比较消息的创建时间（创建消息早的优先）
  (4) 比较消息的ID（ID小的优先）
  ```

  

* $handlePktArrivalEvent$

  ```cpp
  // HomaTransport::ReceiveScheduler::SchedSenders::handlePktArrivalEvent()
  
  // 在每个数据包到达时，接收者视角下的状态（发送者的排名等）可能会发生变化。
  ```

  

* $getBytesOnWire$

  ```cpp
  // HomaPkt::getBytesOnWire(uint32_t numDataBytes, PktType homaPktType)
  
  // 这个函数涉及数据包的字节计算问题，有机会看一下。
  ```

  

1. $OMNeT++$相关

   ```cpp
   1. scheduleAt() // 该函数来安排未来的自消息
   ```

   

2. 相关问题

   ```cpp
   1. 不太清楚模拟时间是如何更新的，但似乎不重要？（如果要了解过度授权就需要看下）
   2. getPrioForMesg() 这个优先级是在干什么？
   ```

   

3. （预留最后一行）

4. 

   
