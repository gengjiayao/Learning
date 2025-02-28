<center><h1>CPP</h1></center>



## 巧用构造函数

```cpp
outboundMsgMap.emplace(std::piecewise_construct, 
                       std::forward_as_tuple(msgId),
                       std::forward_as_tuple(sendMsg, this, msgId, reqUnschedDataVec));

OutboundMessage* outboundMsg = &(outboundMsgMap.at(msgId));
```

- 前者将创建键值对，后者在指定这个值的时候，会自动调用构造函数，因为其中的 $OutBoundMessage$ 有如下构造函数：

```cpp
explicit OutboundMessage(AppMessage* outMsg,
                         SendController* sxController,
                         uint64_t msgId,
                         std::vector<uint16_t> reqUnschedDataVec);
```





## Lambda表达式

### 基本语法

#### 定义

``` 
[captures] (params) specifiers exception -> ret {body}
```



#### 举例

```cpp
int main() {
  int x = 5;
  auto foo = [x](int y) -> int {return x * y};
  cout << foo(8) << endl;
}
```

- 捕获列表
  - 只能捕获非静态、局部变量，这样的设计是因为静态变量与全局变量可以直接用，根本不需要捕获。
- 限定词
  - 默认：$auto$ 类型为 $int\;(int)\;const$
  - 添加限定词 $mutable$：$auto$ 类型为 $int\;(int)$，这时修改捕获变量，在下次调用仍然存在。
  - 如果捕获列表添加引用，则意味着内部修改会影响外部。
  - 简单说就是，里面修改对外面没影响，外面修改对里面也没影响。
- 返回值
  - 可以做隐式的类型转换。
  - 可以省略，由 $return$ 给出。
- 最简单的形式 `[]{}`



### 特殊的捕获方法

- `[this]` ：可以捕获 $this$ ，即获得了对象的作用域。
- `[=]` ：捕获 $lambda$ 表达式定义作用域的全部变量的值。
- `[&]` ：捕获 $lambda$ 表达式定义作用域的全部变量的引用。



### 一些实例

- 函数递归调用

  ```cpp
  // 写法1
  vector<vector<int>> subsets(vector<int>& nums) {
      vector<vector<int>> ans;
      vector<int> path;
  
      auto dfs = [&](this auto&& dfs, int cur) -> void {
          if (cur == nums.size()) {
              ans.push_back(path);
              return;
          }
          // 选
          path.push_back(nums[cur]);
          dfs(cur + 1); 
          path.pop_back();
          // 不选
          dfs(cur + 1); 
      };
  
      dfs(0);
      return ans;
  }
  
  // 写法2 换掉参数内的dfs名称
  vector<vector<int>> subsets(vector<int>& nums) {
      vector<vector<int>> ans;
      vector<int> path;
  
      auto dfs = [&](this auto&& nihao, int cur) -> void {
          if (cur == nums.size()) {
              ans.push_back(path);
              return;
          }
          // 选
          path.push_back(nums[cur]);
          nihao(cur + 1); 
          path.pop_back();
          // 不选
          nihao(cur + 1); 
      };
  
      dfs(0);
      return ans;
  }
  
  
  // 写法3 不写this
  vector<vector<int>> subsets(vector<int>& nums) {
      vector<vector<int>> ans;
      vector<int> path;
  
      auto dfs = [&](auto&& dfs, int cur) -> void {
          if (cur == nums.size()) {
              ans.push_back(path);
              return;
          }
          // 选
          path.push_back(nums[cur]);
          dfs(dfs, cur + 1); 
          path.pop_back();
          // 不选
          dfs(dfs, cur + 1); 
      };
  
      dfs(dfs, 0);
      return ans;
  }
  
  // 写法4 不写auto 效率会慢
  vector<vector<int>> subsets(vector<int>& nums) {
      vector<vector<int>> ans;
      vector<int> path;
  
      function<void(int)> dfs = [&](int cur) -> void {
          if (cur == nums.size()) {
              ans.push_back(path);
              return;
          }
          // 选
          path.push_back(nums[cur]);
          dfs(cur + 1); 
          path.pop_back();
          // 不选
          dfs(cur + 1); 
      };
  
      dfs(0);
      return ans;
  }
  ```






## STL

### vector

- 清空：`clear`



### string

- 删除最后字符：`pop_back`





## 特性

### ranges

- $ranges::sort()$

```cpp
// 默认升序排序，可以使用Lambda表达式自定义排序函数
std::ranges::sort(); 
```




