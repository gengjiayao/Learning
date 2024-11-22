<center><h1>Conweave Learning</h1></center>



### ns3启动流程

- `run.py`：`os.system` 函数执行 `./waf --run 'scratch/network-load-lanbace {config_name}`，`config_name` 为配置文件。
  - `config_name` 一律保存在了 `/mix/output/` 文件夹下的 `config.txt`。
- `network-load-balance.cc`：`main` 函数中 `conf` 读入上方 `config_name` 的参数。（ $Line\;737$ ）
  - `conf` 中有 `FLOW_FILE`、`CC_MODE` 等各种参数。



### 自上而下协议栈

``` 
RdmaClient --> RdmaDriver --> RdmaHw --> QbbNetDevice --> Swich节点发送
```

- `RdmaClientHelper` 获得由 `conf` 读入的各种信息，请求 $rdma$ 服务。
- `RdmaClient::StartApplication` 调用 `rdma-driver.cc` 中的 `RdmaDriver::AddQueuePair`。
- `m_rdma->AddQueuePair` 调用 `rdma-hw.cc` 中 `RdmaHw::AddQueuePair`。 
