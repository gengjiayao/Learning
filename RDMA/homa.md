<center><h1>Homa</h1></center>

## 整体运行逻辑

- $Homa$ 是消息驱动的模拟器
  - $AppMessage$ 是应用层传递的消息，该消息通过处理，处理成 $OutboundMessage$。
  - $OutboundMessage$ 是 $homa$ 层面的消息，进行后续处理和调度。
    - 
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

   
