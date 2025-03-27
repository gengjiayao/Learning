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



- S2. 技巧双指针（更好）

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
// 注意cur为nullptr时，nxt不存在的问题。
// 头插法，cur存在即可插
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode *pre = nullptr;
        ListNode *cur = head;
        while (cur) {
            ListNode *next = cur->next;
            cur->next = pre; // 头插法1 先指ans
            pre = cur; // 头插法2 再指当前
            cur = next;
        }
        return pre;
    }
};
```



### [234. 回文链表](https://leetcode.cn/problems/palindrome-linked-list/)

- S1. 找中点+翻转

```cpp
// 在找中点时放松，比较时严格，按照head2比较
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
        while (head2) { // 也可以写成 head && head2 本题更快
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
    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
        ListNode dummy(-1, nullptr);
        ListNode *pre = &dummy;

        while (list1 && list2) {
            if (list1->val < list2->val) {
                pre->next = list1;
                list1 = list1->next;
            } else {
                pre->next = list2;
                list2 = list2->next;
            }
            pre = pre->next;
        }
        pre->next = list1 ? list1 : list2;
        return dummy.next;
    }
};
```



### [2. 两数相加](https://leetcode.cn/problems/add-two-numbers/)

- 模拟

```cpp
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        ListNode dummy;
        ListNode *pre = &dummy;

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
            pre->next = new ListNode(sum % 10);
            sum /= 10;
            pre = pre->next;
        }
        return dummy.next;
    }
};
```



### [19. 删除链表的倒数第 N 个结点](https://leetcode.cn/problems/remove-nth-node-from-end-of-list/)

- 前后双指针

```cpp
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode dummy(0, head);

        ListNode *fast = &dummy, *slow = &dummy; // 一定指向首节点

        while(n--) {
            fast = fast->next;
        } 

        while (fast->next) {
            slow = slow->next;
            fast = fast->next;
        }
        ListNode *tmp = slow->next;
        slow->next = slow->next->next;

        delete tmp;
        return dummy.next;
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



### [25. K 个一组翻转链表](https://leetcode.cn/problems/reverse-nodes-in-k-group/)

- 哨兵，翻转技巧

```cpp
class Solution {
public:
    ListNode* reverseKGroup(ListNode* head, int k) {
        int n = 0;
        ListNode *p = head;
        while (p) {
            n++;
            p = p->next;
        }

        ListNode dummy;
        dummy.next = head;

        ListNode *p0 = &dummy; // p0是每一段的哨兵
        ListNode *pre = nullptr;
        ListNode *cur = head;
        while (n >= k) {
            n -= k;
            for (int i = 0; i < k; i++) {
                ListNode *nxt = cur->next;
                cur->next = pre;
                pre = cur;
                cur = nxt;
            }
            ListNode *nxt = p0->next; // 记录下一次哨兵应该在的位置

            p0->next->next = cur;
            p0->next = pre;
            p0 = nxt;
        }
        return dummy.next;
    }
};
```



### [138. 随机链表的复制](https://leetcode.cn/problems/copy-list-with-random-pointer/)

- S1. 两次遍历+哈希

```cpp
class Solution {
public:
    unordered_map<Node*, Node*> cache;
    Node* copyRandomList(Node* head) {
        Node *p = head;
        while (p) {
            cache[p] = new Node(p->val);
            p = p->next;
        }
        p = head;
        while (p) {
            cache[p]->next = cache[p->next];
            cache[p]->random = cache[p->random];
            p = p->next;
        }
        return cache[head];
    }
};
```



- S2. 回溯+哈希

```cpp
class Solution {
public:
    unordered_map<Node*, Node*> cache;
    Node* copyRandomList(Node* head) {
        if (head == nullptr) return nullptr;

        if (!cache[head]) {
            Node *newHead = new Node(head->val);
            cache[head] = newHead;
            newHead->next = copyRandomList(head->next);
            newHead->random = copyRandomList(head->random);
        }
        return cache[head];
    }
};
```



- S3. 不哈希 技巧

```cpp
class Solution {
public:
    Node* copyRandomList(Node* head) {
        if (!head) return nullptr;

        Node *cur = head;
        while (cur) {
            Node *tmp = new Node(cur->val);
            tmp->next = cur->next;
            cur->next = tmp;
            cur = tmp->next;
        }

        cur = head;
        while (cur) {
            if (cur->random) 
                cur->next->random = cur->random->next;
            cur = cur->next->next;
        }

        Node *res = head->next;
        Node *pre = head;
        cur = head->next;

        while (cur->next) {
            pre->next = pre->next->next;
            cur->next = cur->next->next;
            pre = pre->next;
            cur = cur->next;
        }
        pre->next = nullptr; // 恢复原链表最后一个节点的next指针为nullptr
        return res;
    }
};
```



### [148. 排序链表](https://leetcode.cn/problems/sort-list/)

- S1. 归并

```cpp
class Solution {
public:
    // 快慢指针slow指向的是第二条链
    ListNode* mid(ListNode* head) {
        // 用pre记录，为了切割
        ListNode *pre = head, *slow = head, *fast = head;
        while (fast && fast->next) {
            pre = slow;
            slow = slow->next;
            fast = fast->next->next;
        }
        pre->next = nullptr;
        return slow;
    }

    ListNode *merge(ListNode* l1, ListNode* l2) {
        ListNode dummy;
        ListNode *cur = &dummy;
        while(l1 && l2) {
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
        return dummy.next;
    }

    ListNode* sortList(ListNode* head) {
        if (!head || !head->next) {
            return head;
        }

        ListNode *head2 = mid(head);

        head = sortList(head);
        head2 = sortList(head2);

        return merge(head, head2);
    }
};
```



### [23. 合并 K 个升序链表](https://leetcode.cn/problems/merge-k-sorted-lists/)

- 分治+合并

```cpp
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        ListNode dummy;
        ListNode* cur = &dummy;

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
        return dummy.next;
    }

    ListNode* merge(vector<ListNode*>& lists, int l, int r) {
        if (l == r) return lists[l];

        int mid = (l + r) >> 1;
        auto left = merge(lists, l, mid);
        auto right = merge(lists, mid + 1, r);
        return mergeTwoLists(left, right);
    }

    ListNode* mergeKLists(vector<ListNode*>& lists) {
        if (lists.size() == 0) return nullptr; // lists为空的情况
        return merge(lists, 0, lists.size() - 1); // 闭区间分治
    }
};
```

- 最小堆

```cpp
class Solution {
public:
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        auto cmp = [](ListNode *a, ListNode *b) {
            return a->val > b->val; // 用大于号构造小堆
        };

        priority_queue<ListNode*, vector<ListNode*>, decltype(cmp)> q;

        for (auto list : lists) {
            if (list) q.push(list);
        }

        ListNode dummy;
        ListNode *cur = &dummy;

        while (!q.empty()) {
            ListNode *tmp = q.top();
            cur->next = tmp;
            cur = cur->next;

            q.pop();
            if (tmp->next) {
                q.push(tmp->next);
            }
        }
        return dummy.next;
    }  
};
```



### [146. LRU 缓存](https://leetcode.cn/problems/lru-cache/)

- S1. 极致 $STL$

```cpp
class LRUCache {
public:
    LRUCache(int cap) : cap (cap) {}

    int get(int k) {
        if (m.count(k)) {
            l.splice(l.begin(), l, m[k]);
            return m[k]->second;
        }
        return -1;
    }
    
    void put(int k, int v) {
        if (m.count(k)) l.erase(m[k]);
        l.push_front({k, v});
        m[k] = l.begin();
        if (l.size() > cap) m.erase(l.back().first), l.pop_back();
    }
private: 
    int cap;
    list<pair<int, int>> l;
    unordered_map<int, decltype(l.begin())> m;
};
```



- S2. 手撕双向循环链表

```cpp
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



## 图论

### [200. 岛屿数量](https://leetcode.cn/problems/number-of-islands/)

- $DFS$（自己写的版本）

```cpp
class Solution {
public:
    int numIslands(vector<vector<char>>& grid) {
        int row = grid.size();
        int col = grid[0].size();

        vector<vector<int>> st(row, vector<int>(col));
        int ans = 0;

        auto dfs = [&] (this auto&& dfs, int i, int j) {
            if (i < 0 || i >= row || j < 0 || j >= col || st[i][j]) return;

            if (grid[i][j] == '1') {
                st[i][j] = 1;
                dfs(i - 1, j);
                dfs(i, j - 1);
                dfs(i + 1, j);
                dfs(i, j + 1);
            }
        };

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (!st[i][j] && grid[i][j] == '1') {
                    ans++;
                    dfs(i, j);
                }
            }
        }
        return ans;
    }
};
```

- S2. $DFS$ 速度更快

```cpp
class Solution {
public:
    int numIslands(vector<vector<char>>& grid) {
        int row = grid.size();
        int col = grid[0].size();

        int ans = 0;
				// 这么写相当于少了边界条件的递归
        auto dfs = [&] (this auto&& dfs, int i, int j) -> void {
            grid[i][j] = '0';
            if (i - 1 >= 0 && grid[i - 1][j] == '1') dfs(i - 1, j);
            if (j - 1 >= 0 && grid[i][j - 1] == '1') dfs(i, j - 1);
            if (i + 1 < row && grid[i + 1][j] == '1') dfs(i + 1, j);
            if (j + 1 < col && grid[i][j + 1] == '1') dfs(i, j + 1);
        };

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (grid[i][j] == '1') {
                    ans++;
                    dfs(i, j);
                }
            }
        }
        return ans;
    }
};
```

- S3. $BFS$ 做法

```cpp

```

- S4. 并查集做法

```cpp

```



## 回溯

### [46. 全排列](https://leetcode.cn/problems/permutations/)

- S1. 选数填（自己写的版本）

```cpp
class Solution {
    vector<vector<int>> ans;
    vector<int> cur;
public:
    void dfs(vector<int>& nums, bool st[], int u) {
        if (u == nums.size()) {
            ans.push_back(cur);
            return;
        }

        for (int i = 0; i < nums.size(); i++) {
            if (!st[i]) {
                cur.push_back(nums[i]);
                st[i] = true;
                dfs(nums, st, u + 1);
                cur.pop_back();
                st[i] = false;
            }
        }
    }

    vector<vector<int>> permute(vector<int>& nums) {
        const int n = nums.size();
        bool st[n];
        fill(st, st + n, false);
        dfs(nums, st, 0);
        return ans;
    }
};
```

- S2. $Lambda$ 表达式版本

```cpp
class Solution {
public:
    vector<vector<int>> permute(vector<int>& nums) {
        const int n = nums.size();
        vector<vector<int>> ans;
        vector<int> path;
        vector<bool> st(n, false);

        auto dfs = [&](this auto&& dfs, int u)  -> void {
            if (u == n) {
                ans.push_back(path);
                return;
            }

            for (int i = 0; i < n; i++) {
                if (!st[i]) {
                    path.push_back(nums[i]);
                    st[i] = true;
                    dfs(u + 1);
                    st[i] = false;
                    path.pop_back();
                }
            }
        };

        dfs(0);
        return ans;
    }
};
```

- S3. 另一种覆盖的方式（速度最快）

```cpp
class Solution {
public:
    vector<vector<int>> permute(vector<int>& nums) {
        const int n = nums.size();
        vector<vector<int>> ans;
        vector<int> path(n); // 直接初始化
        vector<bool> st(n, false);

        auto dfs = [&](this auto&& dfs, int u) -> void {
            if (u == n) {
                ans.push_back(path);
                return;
            }

            for (int i = 0; i < n; i++) {
                if (!st[i]) {
                    path[u] = nums[i]; // 直接覆盖，并且不需要恢复现场
                    st[i] = true;
                    dfs(u + 1);
                    st[i] = false;
                }
            }
        };

        dfs(0);
        return ans;
    }
};
```



### [78. 子集](https://leetcode.cn/problems/subsets/)

- S1. 选或不选：递归

```cpp
class Solution {
public:
    vector<vector<int>> ans;
    vector<int> path;

    void dfs(int cur, vector<int> &nums) {
        if (cur == nums.size()) {
            ans.push_back(path);
            return;
        }
        // 选
        path.push_back(nums[cur]);
        dfs(cur + 1, nums); 
        path.pop_back();
        // 不选
        dfs(cur + 1, nums); 

    }

    vector<vector<int>> subsets(vector<int>& nums) {
        dfs(0, nums);
        return ans;
    }
};
```

- S2. 选或不选：$Lambda$ 表达式

```cpp
class Solution {
public:
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
};
```

- S3. 枚举选哪个

```cpp
class Solution {
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        const int n = nums.size();
        vector<vector<int>> ans;
        vector<int> path;

        auto dfs = [&] (this auto&& dfs, int cur) -> void { // 此时没有return语句，要显示写出返回类型
            ans.push_back(path);
            for (int i = cur; i < n; i++) {
                path.push_back(nums[i]);
                dfs(i + 1);
                path.pop_back();
            }
        };

        dfs(0);
        return ans;
    }
};
```

- S4. 二进制枚举

```cpp
class Solution {
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        const int n = nums.size();
        vector<vector<int>> ans;
        vector<int> path;

        for (int mask = 0; mask < (1 << n); mask++) {
            path.clear();
            for (int i = 0; i < n; i++)
                if (mask & (1 << i)) 
                    path.push_back(nums[i]);
            ans.push_back(path);
        }
    
        return ans;
    }
};
```



### [17. 电话号码的字母组合](https://leetcode.cn/problems/letter-combinations-of-a-phone-number/)

- S1. 组合

```cpp
class Solution {
public:
    vector<string> letterCombinations(string digits) {
        unordered_map<int, string> hash = {
            {2, "abc"}, {3, "def"}, {4, "ghi"}, {5, "jkl"}, 
            {6, "mno"}, {7, "pqrs"}, {8, "tuv"}, {9, "wxyz"}
        };

        string path;
        vector<string> ans;
        const int n = digits.length();

        if (n == 0) return {};
        
        auto dfs = [&](this auto&& dfs, int cur) {
            if (cur == n) {
                ans.push_back(path);
                return;
            }
            string ss = hash[digits[cur] - '0'];
            for (int i = 0; i < ss.length(); i++) {
                path += ss[i];
                dfs(cur + 1);
                path.pop_back();
            }
        };

        dfs(0);
        return ans;
    }
};
```



### [39. 组合总和](https://leetcode.cn/problems/combination-sum/)

- S1. 递归未剪枝

```cpp
class Solution {
public:
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        const int n = candidates.size();

        vector<int> path;
        vector<vector<int>> ans;
        int sum = 0;
        auto dfs = [&] (this auto&& dfs, int cur) {
            if (sum == target) {
                ans.push_back(path);
                return;
            }
            
            if (sum > target || cur == n) return;

            // 选
            sum += candidates[cur];
            path.push_back(candidates[cur]);
            dfs(cur);
            path.pop_back();
            sum -= candidates[cur];

            // 不选
            dfs(cur + 1);
        };

        dfs(0);
        return ans;
    }
};
```

- S2. S1剪枝

```cpp
class Solution {
public:
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        const int n = candidates.size();
        ranges::sort(candidates); // 这个sort更快

        vector<int> path;
        vector<vector<int>> ans;
        int sum = 0;
      
      	// 考虑当前cur位置，选还是不选
        auto dfs = [&] (this auto&& dfs, int cur) {
            if (sum == target) {
                ans.push_back(path);
                return;
            }
            
            // 合并了下面两条语句
            // 注意！cur == n 在前，因为要保证candidates[cur]不越界
            if (cur == n || target - sum < candidates[cur]) return;

            // if (sum > target || cur == n) return;
            // if (target - sum < candidates[cur]) return; // 剪枝

            // 选
            sum += candidates[cur];
            path.push_back(candidates[cur]);
            dfs(cur);
            path.pop_back();
            sum -= candidates[cur];

            // 不选
            dfs(cur + 1);
        };

        dfs(0);
        return ans;
    }
};
```

- S3. 枚举从哪开始选

```cpp
class Solution {
public:
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        const int n = candidates.size();
        ranges::sort(candidates); // 排序后可剪枝

        vector<int> path;
        vector<vector<int>> ans;
        int sum = 0;

        // 选到cur位置了，从这个位置继续看后面
        auto dfs = [&] (this auto&& dfs, int cur) {
            if (sum == target) {
                ans.push_back(path);
                return;
            }
            
            // 合并了下面两条语句
            if (target - sum < candidates[cur]) return;

            // if (sum > target) return;
            // if (target - sum < candidates[cur]) return; // 剪枝

            for (int i = cur; i < n; i++) {
                sum += candidates[i];
                path.push_back(candidates[i]);
                dfs(i); // 这里因为for循环的限制，保证了不会越界，所以不需要判断cur == n的情况
                sum -= candidates[i];
                path.pop_back();
            }
        };

        dfs(0);
        return ans;
    }
};
```

- S4. 完全背包预处理

```cpp
class Solution {
public:
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        const int n = candidates.size();
        
        // dp[i][j] 考虑前i个，target为j，是否有组合方案
        vector dp(n + 1, vector<int>(target + 1));
        dp[0][0] = 1;

        // j从0开始是因为：这是在考虑能不能恰好装满，而不是考虑装满的最大/最小价值
        // 也就是说dp[1][3] = 1 那么dp[2][3]也应该等于1，i = 2的时候不应该限制j的遍历初始值
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= target; j++) {
                if (dp[i][j] || j >= candidates[i] && dp[i + 1][j - candidates[i]])
                    dp[i + 1][j] = 1;
            }
        }
        
        vector<int> path;
        vector<vector<int>> ans;
        int sum = 0;

        auto dfs = [&] (this auto&& dfs, int cur) {
            if (sum == target) {
                ans.push_back(path);
                return;
            }

            if (target - sum >= 0 && dp[cur + 1][target - sum]) {
                // 选
                path.push_back(candidates[cur]);
                sum += candidates[cur];
                dfs(cur);
                sum -= candidates[cur];
                path.pop_back();

                // 不选
                dfs(cur - 1);
            }
        };

        dfs(n - 1); // 倒着递归，因为dp的定义是前i个能否组成target，得在当前向前找
        return ans;
    }
};
```



### [22. 括号生成](https://leetcode.cn/problems/generate-parentheses/)

- S1. 第一次做的回溯思路

```cpp
class Solution {
public:
    vector<string> generateParenthesis(int n) {
        vector<string> ans;
        string path; // 没有初始化，一点点更新出来，速度会慢一点
        auto dfs = [&] (this auto&& dfs, int l, int r) {
            if (l == n && r == n) {
                ans.push_back(path);
                return;
            }

            if (l < n) {
                path += '(';
                dfs(l + 1, r);
                path.pop_back();
            }

            if (r < l) {
                path += ')';
                dfs(l, r + 1);
                path.pop_back();
            }
        };

        dfs(0, 0);
        return ans;
    }
};
```

- S2. S1的速度优化

```cpp
class Solution {
public:
    vector<string> generateParenthesis(int n) {
        vector<string> ans;
        int m = 2 * n;
        string path(m, 0); // 先初始化，后续直接覆盖
        auto dfs = [&] (this auto&& dfs, int l, int r) {
            if (l + r == m) {
                ans.push_back(path);
                return;
            }

            if (l < n) {
                path[l + r] = '(';
                dfs(l + 1, r);
            }

            if (r < l) {
                path[l + r] = ')';
                dfs(l, r + 1);
            }
        };

        dfs(0, 0);
        return ans;
    }
};
```

- S3. 枚举左括号的位置

```cpp
class Solution {
public:
    vector<string> generateParenthesis(int n) {
        vector<string> ans;
        vector<int> path; // 记录左括号位置
        const int m = 2 * n;

        // dfs(i, j) 表示当前填了i个左括号，j个右括号
        auto dfs = [&] (this auto&& dfs, int l, int r) {
            if (l == n) { // 左括号够数就行了，剩下的都给右括号
                string tmp(m, ')');
                for (int p : path) {
                    tmp[p] = '(';
                }
                ans.push_back(tmp);
                return;
            }

            for (int i = 0; i <= l - r; i++) {
                // i 表示目前还可以填的右括号的个数
                path.push_back(l + r + i); // l+r+i 这个位置填一个左括号
                dfs(l + 1, r + i);
                path.pop_back();
            }
        };

        dfs(0, 0);
        return ans;
    }
};
```



### [79. 单词搜索](https://leetcode.cn/problems/word-search/)

- S1. $DFS$ 自己的版本（确定能匹配了再进入搜索）

```cpp
class Solution {
public:
    bool exist(vector<vector<char>>& board, string word) {
        const int row = board.size();
        const int col = board[0].size();

        const int n = word.length();     
        bool flag = false;

        auto dfs = [&] (this auto&& dfs, int i, int j, int cur) {
            if (cur == n) {
                flag = true;
                return;
            } 

            board[i][j] = 0;
            if (i - 1 >= 0 && board[i - 1][j] == word[cur]) dfs(i - 1, j, cur + 1);
            if (j - 1 >= 0 && board[i][j - 1] == word[cur]) dfs(i, j - 1, cur + 1);
            if (i + 1 < row && board[i + 1][j] == word[cur]) dfs(i + 1, j, cur + 1);
            if (j + 1 < col && board[i][j + 1] == word[cur]) dfs(i, j + 1, cur + 1);
            board[i][j] = word[cur - 1];
        };

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (!flag && board[i][j] == word[0]) {
                    dfs(i, j, 1);
                }
            }   
        }
        return flag;
    }
};
```

- S2. S1优化

```cpp
class Solution {
public:
    bool exist(vector<vector<char>>& board, string word) {
        const int row = board.size();
        const int col = board[0].size();

        const int n = word.length();     
        bool flag = false;
				
      	// 优化1: 考察board与word中字符数量
        unordered_map<char, int> cnt;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                cnt[board[i][j]]++;
            }
        }

        unordered_map<char, int> word_cnt;
        for (auto &c : word){
           if (++word_cnt[c] > cnt[c]) {
            return false;
           }
        }
				
        // 优化2: 让word更少的头或者尾开头，匹配的次数更少
        if (cnt[word.back()] < cnt[word[0]]) {
            ranges::reverse(word);
        }

        auto dfs = [&] (this auto&& dfs, int i, int j, int cur) {
            if (cur == n) {
                flag = true;
                return;
            } 

            board[i][j] = 0;
            if (i - 1 >= 0 && board[i - 1][j] == word[cur]) dfs(i - 1, j, cur + 1);
            if (j - 1 >= 0 && board[i][j - 1] == word[cur]) dfs(i, j - 1, cur + 1);
            if (i + 1 < row && board[i + 1][j] == word[cur]) dfs(i + 1, j, cur + 1);
            if (j + 1 < col && board[i][j + 1] == word[cur]) dfs(i, j + 1, cur + 1);
            board[i][j] = word[cur - 1];
        };

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (!flag && board[i][j] == word[0]) {
                    dfs(i, j, 1);
                }
            }   
        }
        return flag;
    }
};
```

- S3. 其他搜索（先进 $dfs$ 再检测是否能匹配）

```cpp
class Solution {
public:
    bool exist(vector<vector<char>>& board, string word) {
        int d[4][2] = {{0, 1}, {0, -1}, {1, 0}, {-1 ,0}};

        const int row = board.size();
        const int col = board[0].size();
        const int n = word.length();     

        unordered_map<char, int> cnt;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                cnt[board[i][j]]++;
            }
        }

        unordered_map<char, int> word_cnt;
        for (auto &c : word){
           if (++word_cnt[c] > cnt[c]) {
            return false;
           }
        }

        if (cnt[word.back()] < cnt[word[0]]) {
            ranges::reverse(word);
        }

        auto dfs = [&] (this auto&& dfs, int i, int j, int cur) -> bool {
            if (board[i][j] != word[cur]) return false;
            if (cur + 1 == n) return true;
            
            board[i][j] = 0;
            for (auto& [dx, dy] : d) {
                int x = i + dx;
                int y = j + dy;
                if (x >= 0 && x < row && y >= 0 && y < col && dfs(x, y, cur + 1)) {
                    return true;
                }
            }
            board[i][j] = word[cur];
            return false;
        };

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if(dfs(i, j, 0)) {
                    return true;
                }
            }   
        }
        return false;
    }
};
```



### [131. 分割回文串](https://leetcode.cn/problems/palindrome-partitioning/)

- S1. 枚举每一个空位

```cpp
class Solution {
public:
    bool findit(string& s, int left, int right) {
        while (left <= right) {
            if (s[left++] != s[right--]) {
                return false;
            }
        }
        return true;
    }

    vector<vector<string>> partition(string s) {
        vector<vector<string>> ans;
        vector<string> path;
        const int n = s.length();

        auto dfs = [&] (this auto&& dfs, int left, int right) {
            if (right == n) {
                ans.push_back(path);
                return;
            }

            // right不是最后一个
            if (right < n - 1) {
                dfs(left, right + 1);
            }

            if (findit(s, left, right)) {
                path.push_back(s.substr(left, right - left + 1));
                dfs(right + 1, right + 1);
                path.pop_back();
            }
        };
        dfs(0, 0);
        return ans;
    }
};
```

- S2. 枚举每一个串的结束位置

```cpp
class Solution {
public:
    bool findit(string& s, int left, int right) {
        while (left <= right) {
            if (s[left++] != s[right--]) {
                return false;
            }
        }
        return true;
    }

    vector<vector<string>> partition(string s) {
        vector<vector<string>> ans;
        vector<string> path;
        const int n = s.length();

        auto dfs = [&] (this auto&& dfs, int start) {
            if (start == n) {
                ans.push_back(path);
                return;
            }

            for (int i = start; i < n; i++) {
                if (findit(s, start, i)) {
                    path.push_back(s.substr(start, i - start + 1));
                    dfs(i + 1);
                    path.pop_back();
                }
            }
            
        };

        dfs(0);
        return ans;
    }
};
```

- S3. 

```cpp
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



### [139. 单词拆分](https://leetcode.cn/problems/word-break/)

- S1. 枚举每个位置，往前看，看能不能划分

```cpp
class Solution {
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        int len = s.length();
        vector<bool> dp(len + 1);
        int wordDict_len = wordDict.size();
        dp[0] = true; // 以实下标结尾的字符串可分
        for (int i = 1; i <= len; i++) { // 实下标
            for (auto &word : wordDict) {
                int word_len = word.length();
                if (i - word_len >= 0 && s.substr(i - word_len, word_len) == word) {
                    dp[i] = dp[i] || dp[i - word_len]; // 类似取最大值
                }
            }
        }
        return dp[len];
    }
};
```

- S2. S1的优化

```cpp
class Solution {
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        const int len = s.length();
        bool dp[len + 1];
        memset(dp, 0, sizeof(dp));
        dp[0] = true; // 以壹起下标，结尾的字符串可分

        for (int i = 1; i <= len; i++) { // 从壹起下标，开始一个一个往前看
            for (string &word : wordDict) {
                if (dp[i]) continue; // 因为是类似取最大值，有就过
                int j = i - word.length();
                if (j >= 0 && s.substr(j, word.length()) == word) {
                    dp[i] = dp[j]; // 从前往后走，可省略dp[i]
                }
            }
        }
        return dp[len];
    }
};
```

- S3. 看每个可划分点，往后推，哪些后边的点是可划分点

```cpp
class Solution {
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        const int len = s.length();
        bool dp[len + 1];
        memset(dp, 0, sizeof(dp)); // fill(dp+1, dp+len+1, false);
        dp[0] = true;

        for (int i = 0; i < len; i++) {
            if (!dp[i]) continue;
            for (string & word : wordDict) {
                int word_len = word.length();
                int j = i + word_len;
                if (j <= len && s.substr(i, word_len) == word) {
                    dp[j] = true;
                }
            }
        }
        return dp[len];
    }
};
```



### [300. 最长递增子序列](https://leetcode.cn/problems/longest-increasing-subsequence/)

- S1. 以 $i$ 结尾的长度（动态规划）

```cpp
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        const int n = nums.size();
        int dp[n];
        memset(dp, 0, sizeof dp);

        for (int i = 0; i < n; i++) {
            dp[i] = 1;
            for (int j = 0; j < i; j++) {
                if (nums[j] < nums[i]) {
                    dp[i] = max(dp[j] + 1, dp[i]);
                }
            }
        }
        int ans = 0;
        for (int i = 0; i < n; i++) 
            ans = max(ans, dp[i]);
        
        return ans;
    }
};
```

- S2. S1小优化

```cpp
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        const int n = nums.size();
        vector<int> dp(n);

        for (int i = 0; i < n; i++) {
            dp[i] = 1;
            for (int j = 0; j < i; j++) {
                if (nums[j] < nums[i]) {
                    dp[i] = max(dp[j] + 1, dp[i]);
                }
            }
        }
        return ranges::max(dp);
    }
};
```

- S3. 长度为 $i + 1$ 的上升序列最小值（贪心+二分）

```cpp
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        vector<int> f;
        for (auto &num : nums) {
            auto it = ranges::lower_bound(f, num);
            if (it == f.end()) f.emplace_back(num);
            else *it = num;
        }
        return f.size();
    }
};
```

- S4. S3的展开

```cpp
class Solution {
public:
    // 找到第一个大于等于target的数
    int lower_bound(const vector<int> &f, const int &target) {
        int l = 0, r = f.size() - 1;
        while (l <= r) {
            int mid = l + (r - l) / 2;
            if (f[mid] < target) 
                l = mid + 1;
            else 
                r = mid - 1;
        }
        return r + 1;
    }    

    int lengthOfLIS(vector<int>& nums) {
        vector<int> f;
        for (auto &num: nums) {
            int pos = lower_bound(f, num);
            if (pos == f.size()) 
                f.push_back(num);
            else 
                f[pos] = num;
        }
        return f.size();
    }
};
```



### [152. 乘积最大子数组](https://leetcode.cn/problems/maximum-product-subarray/)

- S1. 动态规划

```cpp
class Solution {
public:
    int maxProduct(vector<int>& nums) {
        int n = nums.size();
        vector<int> fmax(n + 1), fmin(n + 1);
        fmax[0] = fmin[0] = 1;
        int ans = -1e8;
        for (int i = 0; i < n; i++) {
            int num = nums[i];
            fmax[i + 1] = max({fmax[i] * num, fmin[i] * num, num});
            fmin[i + 1] = min({fmax[i] * num, fmin[i] * num, num});
            ans = max(fmax[i + 1], ans);
        }
        return ans;
    }
};
```

- S2. S1的空间优化

```cpp
class Solution {
public:
    int maxProduct(vector<int>& nums) {
        int fmax = 1, fmin = 1; // 总是向后推进，不需要数组
        int ans = INT_MIN;
        for (int &num: nums) {
            int tmp = fmax;
            fmax = max({fmax * num, fmin * num, num});
            fmin = min({tmp * num, fmin * num, num});
            ans = max(fmax, ans);
        }
        return ans;
    }
};
```



### [416. 分割等和子集](https://leetcode.cn/problems/partition-equal-subset-sum/)

- S1. 01背包（空间优化版）

```cpp
class Solution {
public:
    bool canPartition(vector<int>& nums) {
        int sum = 0;
        for (int &num : nums) 
            sum += num;
        if (sum & 1) 
            return false;
        
        sum = sum >> 1;
        vector<int> dp(sum + 1);
        dp[0] = 1;

        for (int &num: nums) {
            for(int j = sum; j >= num; j--) {
                dp[j] |= dp[j - num];
            }
            if (dp[sum]) return true;
        }        
        return false;
    }
};
```



### [32. 最长有效括号](https://leetcode.cn/problems/longest-valid-parentheses/)

- 动态规划

```cpp
class Solution {
public:
		// 情况分两种：1. ......() 2. .......))
    int longestValidParentheses(string s) {
        int ans = 0, n = s.length();
        vector<int> dp(n);

        for (int i = 1; i < n; i++) {
            if (s[i] == ')') {
                if (s[i - 1] == '(') {
                    dp[i] = 2;
                    if (i - 2 >= 0) { // 注意数组越界！
                        dp[i] += dp[i - 2];
                    }
                } else if (i - dp[i - 1] - 1 >= 0 && s[i - dp[i - 1] - 1] == '(') { // 注意数组越界！
                    dp[i] = dp[i - 1] + 2;
                    if (i - dp[i - 1] - 2 >= 0) { // 注意数组越界！
                        dp[i] += dp[i - dp[i - 1] - 2];
                    }
                }
            }
            ans = max(ans, dp[i]);
        }
        return ans;
    }
};
```

- 模拟栈（分割思想）

```cpp
class Solution {
public:
    int longestValidParentheses(string s) {
        int n = s.length();
        stack<int> stk;
        stk.push(-1);
        int ans = 0;

        for (int i = 0; i < n; i++) {
            if (s[i] == '(') {
                stk.push(i);
            } else {
                stk.pop();
                if (stk.empty()) {
                    stk.push(i); // 此时的i为当前遍历的最右分割点
                } else {
                    ans = max(ans, i - stk.top());
                }
            }
        }
        return ans;
    }
};
```

- 技巧

```cpp
class Solution {
public:
    int longestValidParentheses(string s) {
        int n = s.length();
        int left = 0, right = 0, ans = 0;
        for (int i = 0; i < n; i++) {
            if (s[i] == '(') left++;
            else right++;

            if (left == right) ans = max(ans, left * 2);
            else if (right > left) left = right = 0;
        }
        left = 0, right = 0;
        for (int i = n - 1; i >= 0; i--) {
            if (s[i] == '(') left++;
            else right++;

            if (left == right) ans = max(ans, right * 2);
            else if (right < left) left = right = 0;
        }

        return ans;
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



### [5. 最长回文子串](https://leetcode.cn/problems/longest-palindromic-substring/)

- S1. 动态规划

```cpp
class Solution {
public:
    string longestPalindrome(string s) {
        const int n = s.length();
        int dp[n][n];
        memset(dp, 0, sizeof dp);

        for (int i = 0; i < n; i++) dp[i][i] = 1;

        int l = 0, r = 0, ans = 1;
        for (int i = n - 2; i >= 0; i--) { // 由于更新要用到下一行，所以要倒着走
            for (int j = i + 1; j < n; j++) {
                if (i + 1 == j) {
                    if (s[i] == s[j]) {
                        dp[i][j] = 1;
                        if (j - i + 1 > ans) {
                            ans = j - i + 1;
                            l = i, r = j;
                        } 
                    }
                } else {
                    if (s[i] == s[j] && dp[i + 1][j - 1] == 1) {
                        dp[i][j] = 1;
                        if (j - i + 1 > ans) {
                            ans = j - i + 1;
                            l = i, r = j;
                        }
                    }
                }
            }
        }
        return s.substr(l, r - l + 1);
    }
};
```

- S2. S1优化

```cpp
class Solution {
public:
    string longestPalindrome(string s) {
        const int n = s.length();
        int dp[n][n];
        memset(dp, 0, sizeof dp);
        for (int i = 0; i < n; i++) dp[i][i] = 1;

        int l = 0, ans = 1;
        for (int i = n - 2; i >= 0; i--) { // 由于更新要用到下一行，所以要倒着走
            for (int j = i + 1; j < n; j++) {
                if (s[i] == s[j] && (dp[i + 1][j - 1] || i + 1 == j)) { // 合并两种情况
                    dp[i][j] = 1;
                    if (j - i + 1 > ans) {
                        ans = j - i + 1;
                        l = i;
                    }
                }
            }
        }
        return s.substr(l, ans);
    }
};
```

- S3. 中心扩展算法（主动去暴力查询每个中心能构成的最长串）

```cpp
class Solution {
public:
    int findMaxLen (const string &s, int l, int r) {
        const int n = s.length();
        while (l >= 0 && r < n && s[l] == s[r]) {
            l--;
            r++;
        }
        return r - l - 1;
    }

    string longestPalindrome(string s) {
        int ans = 1, start = 0;
        const int n = s.length();
        for (int i = 0; i < n; i++) {
            int odd = findMaxLen(s, i, i); // 回文串可能是奇数串
            int even = findMaxLen(s, i, i + 1);
            int localMax = max(even, odd);
            if (ans < localMax) {
                ans = localMax;
                start = i - (localMax - 1) / 2; // 要推出这个
            }
        }
        return s.substr(start, ans);
    }
};
```

- S4. $Manacher$ 算法（马拉车）

```cpp
```



### [1143. 最长公共子序列](https://leetcode.cn/problems/longest-common-subsequence/)

- S1. 基本动态规划

```cpp
class Solution {
public:
    int longestCommonSubsequence(string text1, string text2) {
        const int x = text1.length();
        const int y = text2.length();

        int dp[x + 1][y + 1];

        for (int i = 0; i <= x; i++) dp[i][0] = 0;
        for (int j = 0; j <= y; j++) dp[0][j] = 0;

        for (int i = 1; i <= x; i++) {
            for (int j = 1; j <= y; j++) {
                if (text1[i - 1] == text2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = max(dp[i][j - 1], dp[i - 1][j]);
                }
            }
        }
        return dp[x][y];
    }
};
```

- S2. S1空间优化

```cpp
class Solution {
public:
    int longestCommonSubsequence(string text1, string text2) {
        const int x = text1.length();
        const int y = text2.length();

        int dp[y + 1];

        for (int j = 0; j <= y; j++) dp[j] = 0;

        for (int i = 1; i <= x; i++) {
            int pre = 0; // 记录的是dp[i-1][j-1]
            for (int j = 1; j <= y; j++) {
                int tmp = dp[j];
                if (text1[i - 1] == text2[j - 1]) dp[j] = pre + 1;
                else dp[j] = max(dp[j - 1], dp[j]);
                pre = tmp;
            }
        }
        return dp[y];
    }
};
```



### [72. 编辑距离](https://leetcode.cn/problems/edit-distance/)

- S1. 基本动态规划

```cpp
class Solution {
public:
    int minDistance(string s1, string s2) {
        const int len1 = s1.length();
        const int len2 = s2.length();

        int dp[len1 + 1][len2 + 1];
        
        for (int i = 0; i <= len2; i++) dp[0][i] = i;
        for (int i = 0; i <= len1; i++) dp[i][0] = i;

        for (int i = 1; i <= len1; i++) {
            for (int j = 1; j <= len2; j++) {
                if (s1[i - 1] == s2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = min({dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1]}) + 1;
                }
            }
        } 
        return dp[len1][len2];
    }
};
```

- S2. S1空间优化

```cpp
class Solution {
public:
    int minDistance(string s1, string s2) {
        const int len1 = s1.length();
        const int len2 = s2.length();

        int dp[len2 + 1];
        
        for (int i = 0; i <= len2; i++) dp[i] = i;

        for (int i = 1; i <= len1; i++) {
            int pre = dp[0];
            dp[0]++; // 更新下一个，对应对[i]的初始化
            for (int j = 1; j <= len2; j++) {
                int tmp = dp[j];
                if (s1[i - 1] == s2[j - 1]) dp[j] = pre;
              	// dp[j - 1] 已经更新过  代表dp[i][j - 1]
              	// dp[j] 还没有更新  代表dp[i - 1][j]
                // pre 代表dp[i - 1][j - 1]
                else dp[j] = min({dp[j - 1], dp[j], pre}) + 1;
                pre = tmp;
            }
        } 
        return dp[len2];
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

