<center><h1>Leetcode Hot 100</h1></center>



## 矩阵

### [螺旋矩阵](https://leetcode.cn/problems/spiral-matrix/)

#### S1. 模拟

```cpp
class Solution {
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        int row = matrix.size();
        int col = matrix[0].size();
        vector<int> ans;
        int up = 0, down = row - 1, left = 0, right = col - 1;
        while (true) {
            for (int i = left; i <= right; i++)		// 右
                ans.emplace_back(matrix[up][i]);
            if (++up > down) break;

            for (int i = up; i <= down; i++) 		// 下
                ans.emplace_back(matrix[i][right]);
            if (--right < left) break;

            for (int i = right; i >= left; i--)		// 左
                ans.emplace_back(matrix[down][i]);
            if (--down < up) break;

            for (int i = down; i >= up; i--)		// 上
                ans.emplace_back(matrix[i][left]);
            if (++left > right) break;
        }
        return ans;
    }
};
```



#### S2. 经典转弯遍历

```cpp
class Solution {
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        int to_row[4] = {0, 1, 0, -1}; // 转弯定义
        int to_col[4] = {1, 0, -1, 0}; // 转弯定义

        int m = matrix.size();
        int n = matrix[0].size();
        
        vector<int> ans(m * n);
        int i = 0, j = 0, d = 0;
        for (int k = 0; k < m * n; k++) {
            ans[k] = matrix[i][j];
            matrix[i][j] = 2e9;
            int x = i + to_row[d];
            int y = j + to_col[d];
            if (x < 0 || x >= m || y < 0 || y >= n || matrix[x][y] == 2e9) { // 条件特判
                d = (d + 1) % 4;
            }
            i += to_row[d];
            j += to_col[d];
        }
        return ans;
    }
}; 
```



### [旋转图像](https://leetcode.cn/problems/rotate-image/)

#### S1. 上下翻转+对角线翻转

```cpp
class Solution {
public:
    void swap(int &a, int &b) {
        int tmp = a;
        a = b; 
        b = tmp;
    }

    void rotate(vector<vector<int>>& matrix) {
        int n = matrix.size();
				// 上下翻转
        for (int i = 0; i < n / 2; i++) {
            for (int j = 0; j < n; j++) {
                swap(matrix[i][j], matrix[n - i - 1][j]);
            }
        }
				// 左右翻转
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                if (i == j) continue;
                swap(matrix[i][j], matrix[j][i]);
            }
        }
    }
};
```



#### S2. 按层处理

```cpp
class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
        int n = matrix.size();
        // n/2 表示要处理多少层 
        for (int i = 0; i < n / 2; i++) {
            // j = i 表示开始处理那一层的第1个
            // 这一层要处理 n - 2i - 1 个
            // 从 i 开始，就是到n - i - 1 结束
            for (int j = i; j < n - i - 1; j++) {
                // j 是 start
                int &a = matrix[i][j]; // 第一个
                int &b = matrix[j][n - i - 1]; // 第二个
                int &c = matrix[n - i - 1][n - j - 1]; // 第三个
                int &d = matrix[n - j - 1][i]; // 第四个
                int tmp = a;
                a = d;
                d = c;
                c = b;
                b = tmp;
            }
        }
    }
};
```



### [搜索二维矩阵 II](https://leetcode.cn/problems/search-a-2d-matrix-ii/)

#### S1. 排除法

```cpp
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int m = matrix.size();
        int n = matrix[0].size();
        int i = 0, j = n - 1;
        while(j >= 0 && i < m) {
            if (matrix[i][j] > target) {
                j--; // 那一列都大，排除
            } else if (matrix[i][j] < target) {
                i++; // 这一行都小，排除
            } else {
                return true;
            }
        }
        return false;
    }
};
```



## 链表

### [相交链表](https://leetcode.cn/problems/intersection-of-two-linked-lists/)

#### S1. 哈希地址查找

```cpp
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        unordered_set<ListNode *> s;
        ListNode *p = headA;
        while (p) {
            s.insert(p);
            p = p->next;
        }
        p = headB;
        while(p) {
            if (s.find(p) != s.end()) 
                return p;
            p = p->next;
        }
        return nullptr;
    }
};
```



#### S2. 技巧双指针

```cpp
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        ListNode *a = headA, *b = headB;
        while (a != b) {
            a = a ? a->next : headB; // 把结尾的null算在内，防止死循环
            b = b ? b->next : headA; // 把结尾的null算在内，防止死循环
        }
        return a;
    }
};
```



### [反转链表](https://leetcode.cn/problems/reverse-linked-list/)

#### S1. 经典头插法（画好移动顺序）

```cpp
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode *ans = nullptr;
        ListNode *cur = head;
        while (cur) {
            ListNode *next = cur->next;
            cur->next = ans; // 头插法1 先指ans
            ans = cur; // 头插法2 再指当前
            cur = next;
        }
        return ans;
    }
};
```





## 技巧

### [只出现一次的数字](https://leetcode.cn/problems/single-number/)

#### S1. 异或特性

```cpp
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int ans = 0;
        for (auto &i : nums)
            ans ^= i;
        return ans;
    }
};
```

