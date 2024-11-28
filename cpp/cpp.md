<center><h1>CPP</h1></center>



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























