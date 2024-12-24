<center><h1>Leetcode Hot 100</h1></center>



## 矩阵

### [73. 矩阵置零](https://leetcode.cn/problems/set-matrix-zeroes/)

- S1. 记录

```cpp
class Solution {
public:
    void setZeroes(vector<vector<int>>& matrix) {
        int m = matrix.size();
        int n = matrix[0].size();
        
        int *row = new int[m]();
        int *col = new int[n]();

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == 0) {
                    row[i] = col[j] = 1;
                }
            }
        }

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (row[i] || col[j]) {
                    matrix[i][j] = 0;
                }
            }
        }
        delete[] row;
        delete[] col;
    }
};
```



### [54. 螺旋矩阵](https://leetcode.cn/problems/spiral-matrix/)

- S1. 模拟

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



- S2. 经典转弯遍历

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



### [48. 旋转图像](https://leetcode.cn/problems/rotate-image/)

- S1. 上下翻转+对角线翻转

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



- S2. 按层处理

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



### [240. 搜索二维矩阵 II](https://leetcode.cn/problems/search-a-2d-matrix-ii/)

- S1. 排除法

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

### [160. 相交链表](https://leetcode.cn/problems/intersection-of-two-linked-lists/)

- S1. 哈希地址查找

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



- S2. 技巧双指针

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



### [206. 反转链表](https://leetcode.cn/problems/reverse-linked-list/)

- S1. 经典头插法（画好移动顺序）

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



### [234. 回文链表](https://leetcode.cn/problems/palindrome-linked-list/)

- S1. 找中点+翻转

```cpp
class Solution {
public:
  	// 找中点
    ListNode* getMid(ListNode *head) {
        ListNode *fast = head, *slow = head;
        while (fast && fast->next) {
            slow = slow->next;
            fast = fast->next->next;
        }
        return slow;
    }
		// 翻转
    ListNode* reverseL(ListNode *head) {
        ListNode *pre = nullptr, *cur = head;
        while (cur) {
            ListNode *nxt = cur->next;
            cur->next = pre;
            pre = cur;
            cur = nxt;
        }
        return pre;
    }

    bool isPalindrome(ListNode* head) {
        ListNode *head2 = reverseL(getMid(head));
        while (head2) {
            if (head2->val != head->val) {
                return false;
            }
            head2 = head2->next;
            head = head->next;
        }
        return true;
    }
};
```



### [141. 环形链表](https://leetcode.cn/problems/linked-list-cycle/)

- S1. 快慢指针

```cpp
class Solution {
public:
    bool hasCycle(ListNode *head) {
        ListNode *slow = head, *fast = head;
        while (fast && fast->next) {
            fast = fast->next->next;
            slow = slow->next;
            if (fast == slow) return true;
        }
        return false;
    }
};
```



### [142. 环形链表 II](https://leetcode.cn/problems/linked-list-cycle-ii/)

- S1. 快慢指针+技巧

```cpp
class Solution {
public:
    ListNode *detectCycle(ListNode *head) {
        ListNode *slow = head, *fast = head;
        while (fast && fast->next) {
            slow = slow->next;
            fast = fast->next->next;
            if (slow == fast) {
                while (slow != head) {
                    slow = slow->next;
                    head = head->next;
                }
                return slow;
            }
        }
        return nullptr;
    }
};
```



### [21. 合并两个有序链表](https://leetcode.cn/problems/merge-two-sorted-lists/)

- S1. 递归

```cpp
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
        if (!list1) return list2;
        if (!list2) return list1;
        if (list1->val < list2->val) {
            list1->next = mergeTwoLists(list1->next, list2);
            return list1;
        } 
        list2->next = mergeTwoLists(list1, list2->next);
        return list2;
    }
};
```

- S2. 迭代

```cpp
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        ListNode *node = new ListNode;

        ListNode *cur = node;
        while (l1 && l2) {
            if (l1->val < l2->val) {
                cur->next = l1;
                l1 = l1->next;
            } else {
                cur->next = l2;
                l2 = l2->next;
            }
            cur = cur->next;
        }
        cur->next = l1 ? l1 : l2;
        return node->next;
    }
};
```



### [2. 两数相加](https://leetcode.cn/problems/add-two-numbers/)

- 模拟

```cpp
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        ListNode head;
        ListNode *cur = &head;
        int sum = 0;
        while (l1 || l2 || sum) {
            if (l1) {
                sum += l1->val;
                l1 = l1->next;
            }
            if (l2) {
                sum += l2->val;
                l2 = l2->next;
            }
            cur->next = new ListNode(sum % 10);
            cur = cur->next;
            sum = sum >= 10 ? 1 : 0;
        }
        return head.next;
    }
};
```



### [24. 两两交换链表中的节点](https://leetcode.cn/problems/swap-nodes-in-pairs/)

- 递归

```cpp
class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        if (head == nullptr || head->next == nullptr) return head;

        ListNode* newHead = head->next;
        head->next = swapPairs(newHead->next);
        newHead->next = head;
        return newHead;
    }
};
```

- 递推（复杂版）

```cpp
class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        ListNode dummy;
        dummy.next = head;
        ListNode* first = &dummy;
        ListNode* second = head;
        while (second && second->next) {
            first->next = second->next;
            second->next = first->next->next;
            first->next->next = second;
            first = second;
            second = first->next;
        }
        return dummy.next;
    }
};
```

- 递推（简单版）

```cpp
class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        ListNode dummy;
        dummy.next = head;
        ListNode* node0 = &dummy;
        ListNode* node1 = head;
        while (node1 && node1->next) {
            ListNode* node2 = node1->next;
            ListNode* node3 = node2->next;
            node0->next = node2;
            node2->next = node1;
            node1->next = node3;
            node0 = node1;
            node1 = node3;
        }
        return dummy.next;
    }
};
```



### [19. 删除链表的倒数第 N 个结点](https://leetcode.cn/problems/remove-nth-node-from-end-of-list/)

- 双指针

```cpp
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode start;
        start.next = head; 
        ListNode* first = &start;
        ListNode* second = &start;
        while (n--) {
            first = first->next;
        }
        while (first->next) {
            first = first->next;
            second = second->next;
        }
        ListNode* nxt = second->next;
        second->next = second->next->next;
        delete nxt;
        return start.next;
    }
};
```







## 二叉树

### [94. 二叉树的中序遍历](https://leetcode.cn/problems/binary-tree-inorder-traversal/)

- S1. 递归

```cpp
class Solution {
public:
    void solve(TreeNode* root, vector<int> &res) {
        if (root == nullptr) return;
        solve(root->left, res);
        res.emplace_back(root->val);
        solve(root->right,res);
    }

    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> res;
        solve(root, res);
        return res;
    }
};
```

- S2. 迭代

```cpp
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> res;
        stack<TreeNode *> stk;

        while (!stk.empty() || root) {
            while (root) {
                stk.push(root);
                root = root->left;
            }
            root = stk.top();
            stk.pop();
            res.emplace_back(root->val);
            root = root->right;
        } 
        return res;
    }
};
```



### [104. 二叉树的最大深度](https://leetcode.cn/problems/maximum-depth-of-binary-tree/)

- 简单递归

```cpp
class Solution {
public:
    int maxDepth(TreeNode* root) {
        if (!root) return 0;
        return max(maxDepth(root->left), maxDepth(root->right)) + 1;
    }
};
```



### [226. 翻转二叉树](https://leetcode.cn/problems/invert-binary-tree/)

- 简单递归

```cpp
class Solution {
public:
    TreeNode* invertTree(TreeNode* root) {
        if (!root) return nullptr;
        TreeNode* tmp = root->right;
        root->right = root->left;
        root->left = tmp;
        invertTree(root->left);
        invertTree(root->right);
        return root;
    }
};
```



### [101. 对称二叉树](https://leetcode.cn/problems/symmetric-tree/)

- 简单递归

```cpp
class Solution {
public:
    bool solve(TreeNode* left, TreeNode* right) {
        if (left == nullptr || right == nullptr)
            return left == right;
        return left->val == right->val && solve(left->right, right->left) && solve(left->left, right->right);
    }

    bool isSymmetric(TreeNode* root) {
        return solve(root->left, root->right);
    }
};
```



### [543. 二叉树的直径](https://leetcode.cn/problems/diameter-of-binary-tree/)

- 结合树的深度

```cpp
class Solution {
public:
    int solve (TreeNode* root, int &ans) {
        if (!root) return 0;
        int left_depth = solve(root->left, ans);
        int right_depth = solve(root->right, ans);
        ans = max(left_depth + right_depth + 1, ans);
        return max(left_depth, right_depth) + 1;

    }

    int diameterOfBinaryTree(TreeNode* root) {
        int ans = 0;
        solve(root, ans);
        return ans - 1;
    }
};
```



### [102. 二叉树的层序遍历](https://leetcode.cn/problems/binary-tree-level-order-traversal/)

- 层序遍历

```cpp
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        if (!root) return {}; // 注意
        vector<vector<int>> ans;
        queue<TreeNode*> q;
        q.push(root);
        while(!q.empty()) {
            vector<int> layer;
            int n = q.size();
            while(n--) {
                auto tmp = q.front();
                layer.push_back(tmp->val);
                q.pop();
                if (tmp->left) q.push(tmp->left); // 注意
                if (tmp->right) q.push(tmp->right); // 注意
            }
            ans.emplace_back(layer);
        }
        return ans;
    }
};
```



### [108. 将有序数组转换为二叉搜索树](https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/)

- 闭区间写法

```cpp
class Solution {
public:
    TreeNode* solve(vector<int> &nums, int left, int right) {
        if (left > right) return nullptr;
        int mid = left + (right - left) / 2;
        return new TreeNode(nums[mid], solve(nums, left, mid - 1), solve(nums, mid + 1, right));
    }

    TreeNode* sortedArrayToBST(vector<int>& nums) {
        return solve(nums, 0, nums.size() - 1);
    }
};
```



- 左闭右开区间写法

```cpp
class Solution {
public:
    TreeNode* solve(vector<int> &nums, int left, int right) {
        if (left == right) return nullptr;
        int mid = left + (right - left) / 2;
        return new TreeNode(nums[mid], solve(nums, left, mid), solve(nums, mid + 1, right));
    }

    TreeNode* sortedArrayToBST(vector<int>& nums) {
        return solve(nums, 0, nums.size());
    }
};
```



### [98. 验证二叉搜索树](https://leetcode.cn/problems/validate-binary-search-tree/)

- 简单递归+注意范围+开区间

```cpp
class Solution {
public:
    bool solve (TreeNode* root, long long down, long long up) {
        if (!root) return true;
        bool j3 = root->val > down && root->val < up;
        bool j1 = solve(root->left, down, root->val);
        bool j2 = solve(root->right, root->val, up);
        return j3 && j1 && j2;
    } 

    bool isValidBST(TreeNode* root) {
        return solve(root, LLONG_MIN, LLONG_MAX); // 双开区间
    }
};
```



### [230. 二叉搜索树中第 K 小的元素](https://leetcode.cn/problems/kth-smallest-element-in-a-bst/)

- 递归

```cpp
class Solution {
public:
    void solve (TreeNode* root, int &k, int &ans) {
        if (!root) return;
        solve(root->left, k, ans);
        if (--k == 0) ans = root->val;
        solve(root->right, k, ans);
    }

    int kthSmallest(TreeNode* root, int k) {
        int ans = 0;
        solve(root, k, ans);
        return ans;
    }
};
```

- 迭代

```cpp
class Solution {
public:
    int kthSmallest(TreeNode* root, int k) {
        stack<TreeNode*> stk;
        TreeNode* tmp_node = root;
        
        while (!stk.empty() || tmp_node) {
            while (tmp_node) { // 不断放左孩子
                stk.push(tmp_node);
                tmp_node = tmp_node->left;
            }
            tmp_node = stk.top();
            stk.pop();
            k--;
            if (k == 0) break;
            tmp_node = tmp_node->right; // 重新标定一个根，准备放当前跟的全部左孩子
        }
        return tmp_node->val;
    }
};
```



## 栈

### [20. 有效的括号](https://leetcode.cn/problems/valid-parentheses/)

- S1. 简单的匹配

```cpp
class Solution {
public:
    bool isValid(string s) {
        unordered_map<char, char> mp = {{')', '('}, {']', '['}, {'}', '{'}};

        stack<char> stk;
        for (char &c : s) {
            if (!mp.contains(c)) {
                stk.push(c);
            } else {
                if (!stk.empty() && stk.top() == mp[c]) {
                    stk.pop();
                } else return false;
            }
        }
        return stk.empty();
    }
};
```



## 贪心

### [121. 买卖股票的最佳时机](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/)

- S1. 简单贪心

```cpp
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int min_price = INT_MAX; // 记录当前枚举卖出价格之前的最小价格
        int ans = 0;
        for (int &p : prices) {
            if (p > min_price) {
                ans = max(p - min_price, ans);
            } else {
                min_price = p;
            }
        }
        return ans;
    }
};
```



### [55. 跳跃游戏](https://leetcode.cn/problems/jump-game/)

- S1. 贪心写法1

```cpp
class Solution {
public:
    bool canJump(vector<int>& nums) {
        int n = nums.size();
        int len = 0;
        for (int i = 0; i < n; i++) {
            if (i > len) return false;
            len = max(len, i + nums[i]);
        }
        return true;
    }
};
```



- S2. 贪心写法2

```cpp
class Solution {
public:
    bool canJump(vector<int>& nums) {
        int n = nums.size();
        int len = 0;
        for (int i = 0; i < n && i <= len; i++) {
            len = max(len, i + nums[i]);
        }
        return len >= n - 1;
    }
};
```



### [45. 跳跃游戏 II](https://leetcode.cn/problems/jump-game-ii/)

- S1. 简单贪心

```cpp
class Solution {
public:
    int jump(vector<int>& nums) {
        int cur_end = 0, max_end = nums[0];
        int n = nums.size();
        int ans = 0;
        for (int i = 0; i < n; i++) {
            if (cur_end >= n - 1) break; // 能到达，不用再走了
            max_end = max(i + nums[i], max_end);
            if (i == cur_end) {
                cur_end = max_end;
                ans++;
            }
        }
        return ans;
    }
};
```



### [763. 划分字母区间](https://leetcode.cn/problems/partition-labels/)

- S1. 简单贪心

```cpp
class Solution {
public:
    vector<int> partitionLabels(string s) {
        vector<int> ans;
        int last[26];
        int n = s.length();
        for (int i = 0; i < n; i++) {
            last[s[i] - 'a'] = i;
        }

        int start = 0, end = 0;
        for (int i = 0; i < n; i++) {
            end = max(end, last[s[i] - 'a']);
            if (i == end) {
                ans.emplace_back(end - start + 1);
                start = end + 1;
            }
        }
        return ans;
    }
};
```



## 动态规划

### [70. 爬楼梯](https://leetcode.cn/problems/climbing-stairs/)

- S1. 简单动态规划

```cpp
class Solution {
public:
    int climbStairs(int n) {
        int f[n + 1];
        f[1] = 1;
        f[0] = 1;

        for (int i = 2; i <= n; i++) {
            f[i] = f[i - 1] + f[i - 2];
        }
        return f[n];
    }
};
```



### [118. 杨辉三角](https://leetcode.cn/problems/pascals-triangle/)

- S1. 简单动态规划

```cpp
class Solution {
public:
    vector<vector<int>> generate(int numRows) {
        vector<vector<int>> ans(numRows);
        for (int i = 0; i < numRows; i++) {
            ans[i].resize(i + 1, 1);
            for (int j = 1; j < i; j++) {
                ans[i][j] = ans[i - 1][j] + ans[i - 1][j - 1];
            }
        }
        return ans;
    }
};
```



### [198. 打家劫舍](https://leetcode.cn/problems/house-robber/)

- S1. 简单动态规划

```cpp
class Solution {
public:
    int rob(vector<int>& nums) {
        int n = nums.size();
        vector<int> dp(n);
        if (n == 1) return nums[0];
        dp[0] = nums[0];
        dp[1] = max(nums[0], nums[1]);
        for (int i = 2; i < n; i++) {
            dp[i] = max(dp[i - 2] + nums[i], dp[i - 1]);
        }
        return dp[n - 1];
    }
};
```



### [279. 完全平方数](https://leetcode.cn/problems/perfect-squares/)

- S1. 完全背包

```cpp
class Solution {
public:
    int numSquares(int n) {
        vector<int> f(n + 1, n + 1); // 第2个n+1代表初始化的最大值，因为要恰好装满背包
        f[0] = 0; // 当背包容量为0的时候，装好背包需要的价值是合法的，是0
        for (int i = 1; i * i <= n; i++) {
            for (int j = i * i; j <= n; j++) { // 从背包能装得下的i*i开始遍历
                f[j] = min(f[j - i * i] + 1, f[j]);
            }
        }
        return f[n];
    }
};
```



### [322. 零钱兑换](https://leetcode.cn/problems/coin-change/)

- S1. 完全背包

```cpp
class Solution {
public:
    int coinChange(vector<int>& coins, int amount) {
        vector<int> f(amount + 1, INT_MAX / 2);
        f[0] = 0;
        for (int &x : coins) {
            for (int j = x; j <= amount; j++) {
                f[j] = min(f[j], f[j - x] + 1);
            }
        }
        int ans = f[amount];
        return ans < INT_MAX / 2 ? ans : -1;
    }
};
```



## 多维动态规划

### [62. 不同路径](https://leetcode.cn/problems/unique-paths/)

- S1. 简单动态规划

```cpp
class Solution {
public:
    int uniquePaths(int m, int n) {
        vector<vector<int>> f(m, vector<int>(n));
        for (int j = 0; j < n; j++) f[0][j] = 1;
        for (int i = 0; i < m; i++) f[i][0] = 1;

        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                f[i][j] = f[i - 1][j] + f[i][j - 1];
            }
        }
        return f[m - 1][n - 1];
    }
};
```



- S2. 空间优化

```cpp
class Solution {
public:
    int uniquePaths(int m, int n) {
        vector<int> f(n); // 从左到右，只需要一维数组就可以解决
        for (int j = 0; j < n; j++) f[j] = 1;

        for (int i = 1; i < m; i++)
            for (int j = 1; j < n; j++) 
                f[j] = f[j] + f[j - 1];
        return f[n - 1];
    }
};
```



### [64. 最小路径和](https://leetcode.cn/problems/minimum-path-sum/)

- S1. 简单动态规划

```cpp
class Solution {
public:
    int minPathSum(vector<vector<int>>& grid) {
        int m = grid.size();
        int n = grid[0].size();
        vector<vector<int>> f(m, vector<int>(n));

        f[0][0] = grid[0][0];
        for (int j = 1; j < n; j++) {
            f[0][j] = f[0][j - 1] + grid[0][j];
        }

        for (int i = 1; i < m; i++) {
            f[i][0] = f[i - 1][0] + grid[i][0];
        }

        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                f[i][j] = min(f[i - 1][j], f[i][j - 1]) + grid[i][j];
            }
        }
        return f[m - 1][n - 1];
    }
};
```



- 空间优化

```cpp
class Solution {
public:
    int minPathSum(vector<vector<int>>& grid) {
        int m = grid.size();
        int n = grid[0].size();
        vector<int> f(n);

        f[0] = grid[0][0];
        for (int j = 1; j < n; j++) 
            f[j] = f[j - 1] + grid[0][j];

        for (int i = 1; i < m; i++) 
            for (int j = 0; j < n; j++) 
                if (j == 0) f[0] += grid[i][0];
                else f[j] = min(f[j], f[j - 1]) + grid[i][j];
                
        return f[n - 1];
    }
};
```



## 技巧

### [136. 只出现一次的数字](https://leetcode.cn/problems/single-number/)

- S1. 异或特性

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

