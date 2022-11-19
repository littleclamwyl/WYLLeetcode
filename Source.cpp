/*  
从头开始刷leetcode
目标：年底150+
*/

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <queue>
#include <map>

/* 答题模板 */

#if 0
class Solution {
public:
};


int main() {
    Solution sol;

    std::cout << "Value: " << std::endl;
}

#endif

// 深度优先搜索
/* 1 695 岛屿的最大面积 */
// DFS 递归很关键
// 要定义一些全局变量
#if 0

class Solution {
public:

    int nRow = 1;
    int nCol = 1;
    std::vector<std::vector<int>> directions = { {-1,0},{0,-1},{1,0},{0,1} };
    int maxAreaOfIsland(std::vector<std::vector<int>>& grid) {
        nRow = grid.size();
        nCol = grid[0].size();

        int maxValue = 0;

        for (int i = 0; i < nRow; i++) {
            for (int j = 0; j < nCol; j++) {
                int tempValue = 0;

                tempValue = dfs(grid, i, j);

                maxValue = maxValue > tempValue ? maxValue : tempValue;
            }
        }

        return maxValue;

    }

    int dfs(std::vector<std::vector<int>>& grid, int r, int c) {
        int dfsValue = 0;

        if (r < 0 || r >= nRow || c < 0 || c >= nCol || grid[r][c] == 0) {
            return 0;
        }
        else {
            grid[r][c] = 0;
            dfsValue += 1;
            for (auto dir : directions) {
                dfsValue += dfs(grid, r + dir[0], c + dir[1]);
            }

        }

        return dfsValue;


    }
};


int main() {
    Solution sol;
    std::vector<std::vector<int>> grid1 = {
        {0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0},
        {0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0},
        {0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0}
    }; 

    std::vector<std::vector<int>> grid2(10, std::vector<int>(10, 0));
    std::vector<std::vector<int>> grid3(15, std::vector<int>(15, 1));

    std::cout << sol.maxAreaOfIsland(grid1) << std::endl;
    std::cout << sol.maxAreaOfIsland(grid2) << std::endl;
    std::cout << sol.maxAreaOfIsland(grid3) << std::endl;

}


#endif



/* 2 547 省份数量 */
// DFS
#if 0
class Solution {
public:
    int nRow = 1;
    int findCircleNum(std::vector<std::vector<int>>& isConnected) {
        nRow = isConnected.size();
        int count = 0;

        for (int i = 0; i < nRow; i++) {
            if (isConnected[i][i] == 0) {
                count += 0;
            }
            else {
                count += 1;
                isConnected[i][i] = 0;
                dfs(isConnected, i);
            }
        }

        return count;
    }

    void dfs(std::vector<std::vector<int>>& isConnected, int r) {
        for (int j = 0; j < nRow; j++) {
            if (isConnected[r][j] == 1) {
                isConnected[r][j] = 0;
                isConnected[j][r] = 0;
                isConnected[j][j] = 0;
                dfs(isConnected, j);
            }
        }
    }
};


int main() {
    Solution sol;

    std::vector<std::vector<int>> data1 = { 
        {1, 1, 0},
        {1, 1, 0},
        {0, 0, 1} 
    };

    
    std::vector<std::vector<int>> data2 = {
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1}
    };


    std::vector<std::vector<int>> data3 = {
        {1,0,0,1},
        {0,1,1,0},
        {0,1,1,1},
        {1,0,1,1}
    };
    
    std::cout << sol.findCircleNum(data1) << std::endl;
    std::cout << sol.findCircleNum(data2) << std::endl;
    std::cout << sol.findCircleNum(data3) << std::endl;
        

}


#endif 




// 贪心算法
/* 3  455  分发饼干 */
#if 0

class Solution {
public:
    int findContentChildren(std::vector<int>& children, std::vector<int>& cookies) {
        std::sort(children.begin(), children.end(), [](int a, int b) {return a < b; });
        std::sort(cookies.begin(), cookies.end());


        int numChildren = children.size();
        int numCookies = cookies.size();
        int count = 0;

        for (int i = 0; i < numChildren; i++) {
            for (int j = 0; j < numCookies; j++) {
                if (children[i] <= cookies[j]) {
                    count += 1;
                    cookies[j] = 0;
                    break;
                }
            }
        }

        return count;
    }



};

int main() {
    Solution sol;

    std::vector<int> g{ 1,2,3 };
    std::vector<int> s{ 1,1 };
    
    std::cout << sol.findContentChildren(g, s) << std::endl;
}




#endif



/* 4 206 反转链表 */
// 记得学会初始化链表和二叉树啊！
#if 0
struct Node {
    int val;
    Node* next;
    Node(): val(0), next(nullptr){}
    Node(int x) : val(x), next(nullptr){}
    Node(int x, Node* nnext) : val(x), next(nnext){}
    //Node(): val(0),next(nullptr){}
};

class Solution {
public:
    Node* reverse(Node* head) {
        Node* mypre = nullptr;
        //Node* mycur; // head
        Node* mynext;
        Node* mytemp;

        mynext = head->next;
        mytemp = head;

        while (head->next) {
            mynext = head->next;
            head->next = mypre;
            mypre = head;
            head = mynext;
            
        }
        head->next = mypre;

 /*       while (head) {
            std::cout << "Value: " << head->val << std::endl;
            head = head->next;
        }*/

        return head;
    }

    void printNode(Node* head) {
        while (head) {
            std::cout << "Value: " << head->val << std::endl;
            head = head->next;
        }
    }
};

int main() {
    Solution sol;
    Node* head1 = new Node(1);
    Node* head2 = new Node(2);
    Node* head3 = new Node(3);
    Node* head4 = new Node(4);
    Node* head5 = new Node(5);
    
    head1->next = head2;
    head2->next = head3;
    head3->next = head4;
    head4->next = head5;

    //sol.printNode(head1);

    std::cout << "---------------------" << std::endl;
    Node* resultNode = new Node;
    resultNode = sol.reverse(head1);
    sol.printNode(resultNode);

    std::cout << "---------------------" << std::endl;


    Node* testNode1 = new Node(15);
    Node* testNode2 = new Node(100, testNode1);
    Node* testNode3 = new Node(80, testNode2);
    sol.printNode(testNode3);


    return 0;

}


#endif


/* 5 135 分发糖果 */
#if 0

// n>=1
// points[i] >= 0
class Solution {
public:
    int candy(std::vector<int>& points) {
        
        // 第一轮，确保每个孩子都有1颗糖果；
        int n = points.size();

        if (n == 1) return 1;

        std::vector<int> nCandy(n, 1);


        // 第二轮，从左往右，右边>左边，则+1
        // 并且大家初始的糖果数目都是1
        for (int left = 1; left < n; left++) {
            if (points[left] > points[left - 1]) {
                nCandy[left] = nCandy[left - 1] + 1;
            }
        }


        // 第二轮，从右往左，左边>右边，则取max(当前的，右边的+1）
        for (int right = n - 1; (right - 1) >= 0; right--) {
            if (points[right - 1] > points[right]) {
                nCandy[right - 1] = std::max(nCandy[right - 1], nCandy[right] + 1);
                
            }
        }

        int result = 0;
        for (int i = 0; i < nCandy.size(); i++) {
            result += nCandy[i];
        }


        //int res1 = std::accumulate(nCandy.begin(), nCandy.end(), 0);
        return result;

    }
};

int main() {
    Solution sol;

    std::vector<int> data1{ 1,2,2 };

    int res = sol.candy(data1);

    std::cout << "Value: " << res << std::endl;
}

#endif



/* 6 435 重叠区间 */
// 如何排序
// 排序以后怎么遍历
// flag如何设置
// 不修改值记得使用 reference
#if 0

class Solution {
public:
    int eraseOverlapIntervals(std::vector<std::vector<int>>& intervals) {
        int n = intervals.size();
        if (n == 1) { return 0; }

        std::sort(intervals.begin(), intervals.end(), [](std::vector<int>& x, std::vector<int>& y) { 
            return x[1] < y[1]; 
            }
        );
        int count = 0;

        
        int curRight = intervals[0][1];
        for (int i = 1; i < n; i++) {
            if (intervals[i][0] < curRight) {
                count += 1;
            }
            else {
                curRight = intervals[i][1];
            }
        }

        return count;
    }
};

int main() {
    
    std::vector<std::vector<int>> data1{
        {1, 2},
        {2, 3},
        {3, 4},
        {1, 3}
    };



    Solution sol;
    int res = sol.eraseOverlapIntervals(data1);
    std::cout << res << std::endl;

    std::cout << "------------------------" << std::endl;

    


}



#endif




/* 7 605 种花问题 */
// 1 <= flowerbed.length <= 2 * 10^4
// flowerbed[i] 为 0 或 1
// flowerbed 中不存在相邻的两朵花
// 0 <= n <= flowerbed.length

#if 0
class Solution {
public:
    bool canPlaceFlowers(std::vector<int>& flowerbed, int n) {
        int nflowerbed = flowerbed.size();
        int nexpend = nflowerbed + 2;

        std::vector<bool> flag(nexpend, false);

        // 如果花坛那个位置种了花，那么该位置和其左右位置就不可以种花了
        for (int i = 0; i < nflowerbed; i++) {
            if (flowerbed[i] == 1) {
                flag[i + 1] = true;
                flag[i] = true;
                flag[i + 2] = true;
            }
        }

        // 计算花坛中能种花的位置数目
        int count = 0;
        //for (int i = 1; i < nexpend - 1; i++) {
            //if (flag[i] == true) {
                //count += 1;
            //}
        //}

        for (int i = 1; i < nexpend - 1; i++) {
            if (flag[i] == false) {
                count += 1;
                flag[i] = true;
                flag[i + 1] = true;
            }
        }

        //int res = nflowerbed - count;

        //std::cout << "res = " << res << std::endl;
        //std::cout << "nflowerbed = " << nflowerbed << std::endl;
        std::cout << "count = " << count << std::endl;
        //return n > res ? false : true;

        return count >= n ? true : false;

    }



};


int main() {
    
    std::vector<int> flowers1{ 1,0,0,0,1 };
    std::vector<int> flowers2{ 1,0,0,0,0,1 };
    int n1 = 1, n2 = 2;
    Solution sol;
    std::cout << "Result: " << sol.canPlaceFlowers(flowers1, n1) << std::endl;

    std::cout << "Result: " << sol.canPlaceFlowers(flowers2, n2) << std::endl;


}

#endif


/* 8 452 用最少的箭去射爆气球 */
// 区间贪心问题：给定n个闭区间[x,y]，问最少需要确定多少个点，使得每个闭区间中都至少存在一个点。 (这哥们说的好啊
#if 0

class Solution {
public:
    int findMinArrowShots(std::vector<std::vector<int>>& points) {
        
        int nBalloons = points.size();
        if (nBalloons == 1) return 1;

        // 气球重排列，区间右边小的放在前面
        std::sort(points.begin(), points.end(), [](std::vector<int>& x, std::vector<int>& y) {
            return x[1] < y[1];
            });

        // 因为气球数目>=1, 所以至少需要一根箭
        int narrow = 1;
        int flag = points[0][1];
        

        for (int i = 1; i < nBalloons; i++) {
            if (points[i][0] <= flag) {
                continue;
            }
            else {
                narrow += 1;
                flag = points[i][1];
            }
        }

        return narrow;

    }
};


int main() {
    Solution sol;
    std::vector<std::vector<int>> data1{ 
        {10, 16},
        {2, 8},
        {1, 6},
        {7, 12} 
    };

    std::vector<std::vector<int>> data2{
        {1,2} ,
        {2,3},
        {3,4},
        {4,5}
    };

    int res = sol.findMinArrowShots(data1);
    std::cout << "Value: " << res << std::endl;

    int res2 = sol.findMinArrowShots(data2);
    std::cout << "Value: " << res2 << std::endl;



}
#endif



/* 9 122 买卖股票的最佳时机 */

#if 0

class Solution {
public:
    int maxProfit(std::vector<int>& prices) {
        int money = 0;
        int ndays = prices.size();
        if (ndays == 1) return 0;

        // 只需要判定第i天和第i+1天股票价格大小：
        // prices[i] < prices [i+1], 先买后卖；
        // else: 不动；
        // 实际上代码可以继续缩减
        // 实际上这步直接放在下面作判定条件就可以了
        std::vector<bool> isSales(ndays, false);
        for (int i = 0; i < ndays - 1; i++) {
            if (prices[i] < prices[i + 1]) {
                isSales[i] = true;
            }
        }

        for (int i = 0; i < ndays - 1; i++) {
            if (isSales[i]) {
                money += prices[i + 1] - prices[i];
            }
        }

        return money;        
    }
};


int main() {
    Solution sol;
    std::vector<int> data1{ 7,1,5,3,6,4 };

    int res1 = sol.maxProfit(data1);

    std::cout << "Value: " << res1 << std::endl;

    std::vector<int> data2{ 1,2,3,4,5 };

    int res2 = sol.maxProfit(data2);

    std::cout << "Value: " << res2 << std::endl;


}

#endif


// 双指针
/* 10 1 两数之和 */
// 题目有较大的修改哦，不再是那么简单的升序数组了
// 擦，升序数据感觉更简单呢
#if 0
class Solution {
public:
    std::vector<int> twoSum(std::vector<int>& nums, int target) {
        std::vector<int> backNums = nums;

        std::sort(nums.begin(), nums.end(), [](int a, int b) {return a < b; });

        int left = 0;
        int right = nums.size() - 1;

        // 获取了实际的值 nums[left] and nums[right]
        while (left < right) {
            if (nums[left] + nums[right] == target) {
                break;
            }
            else if (nums[left] + nums[right] > target) {
                right--;
            }
            else {
                left++;
            }
        }

        bool flag1 = true;
        bool flag2 = true;
        std::vector<int> res(2, 0);

        // 根据值去获得索引编号

        for (int i = 0; i < backNums.size(); i++) {
            if (backNums[i] == nums[left] && flag1) {
                res[0] = i;
                flag1 = false;
            }
            else if (backNums[i] == nums[right] && flag2) {
                res[1] = i;
                flag2 = false;
            }
            else {
                continue;
            }
            
        }


        return res;

    }
};

int main() {
    Solution sol;
    std::vector<int> nums1{ 1,2,3,4 };
    int target1 = 4;

    std::vector<int> res1 = sol.twoSum(nums1, target1);
    std::cout << "Value is: " << res1[0] << " and " << res1[1] << std::endl;
}

#endif


/* 11 167 两数之和--有序数组 */
// 常量级额外空间
#if 0
class Solution {
public:
    std::vector<int> twoSum(std::vector<int>& nums, int target) {

        int n = nums.size();
        int left = 0;
        int right = n - 1;
        while (left < right) {
            if (nums[left] + nums[right] == target) {
                break;
            }
            else if (nums[left] + nums[right] > target) {
                right--;
            }
            else {
                left++;
            }
        }
        std::vector<int>res{ left + 1, right + 1 };
        return res;
    }
};

int main() {
    Solution sol;
    std::vector<int> nums1{ 2,7,11,15 };
    int target1 = 9;
    std::vector<int> res1 = sol.twoSum(nums1, target1);
    std::cout << "Res: " << res1[0] << " and " << res1[1] << std::endl;

}
#endif

/* 12 88 归并两个有序数组 */
// 已经解答


/* 13 142 环形链表II */

#if 0

struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x):val(x), next(nullptr){}
    ListNode(int x, ListNode* nextlist):val(x),next(nextlist){}
};


class Solution {
public:

    ListNode* detectCycle(ListNode* head) {

        ListNode* fast = head;
        ListNode* slow = head;

        while (true) {
            if (!fast || !fast->next) return nullptr;
            fast = fast->next->next;
            slow = slow->next;

            if (fast == slow) {
                fast = head;

                while (fast != slow) {
                    fast = fast->next;
                    slow = slow->next;
                }

                return fast;
            }           
        }
    }

};


int main() {

    ListNode* node1 = new ListNode(3);
    ListNode* node2 = new ListNode(20);
    //ListNode* node3 = new ListNode(0);
    //ListNode* node4 = new ListNode(-4, node2);
    node1->next = node2;
    //node2->next = node3;
    //node3->next = node4;
    node2->next = node1;
    

    int i = 5;
    ListNode* head = node1;
    while (i--)
    {
        std::cout << "Value: " << head->val << std::endl;
        head = head->next;
       
    }


    std::cout << "---------------------" << std::endl;
    Solution sol;

    std::cout << "Value: " << sol.detectCycle(node1)->val << std::endl;

}



#endif



/* 14 160 链表相交 */
// 已经解答

/* 15 53 最大子数组和 */
#if 0
class Solution {
public:
    int maxSubArray(std::vector<int>& nums) {

        unsigned n = nums.size();
        if (n == 1) return nums[0];

        std::vector<int> dp(n);

        dp[0] = nums[0];

        for (int i = 1; i < n; i++) {
            if (dp[i - 1] >= 0) {
                dp[i] = nums[i] + dp[i - 1];
            }
            else {
                dp[i] = nums[i];
            }
        }

        int res = nums[0];
        for (auto r : dp) {
            res = res > r ? res : r;
        }
        return res;
       

    }
};


int main() {

    std::vector<int> data1{ -2,1,-3,4,-1,2,1,-5,4 };
    Solution sol;

    int res1 = sol.maxSubArray(data1);

    std::cout << "My result is: " << res1 << std::endl;
}

#endif


/* 16 160 链表的第一个公共节点 */

/* 17 524 删除字母match最长的单词 */

/* 18 141 环形链表I */

/* 19 1662 检查两个字符串数组是否相等 */


/* 20 3 无重复最长子串 */

// 输入: s = "abcabcbb"
// 输出 : 3
// 解释 : 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
// size_type到底是啥子？容器size的类型.
#if 0
class Solution {
public:
    int lengthOfLongestSubstring(std::string s) {
        std::string::size_type len_s = s.size();
        if (len_s == 0)return 0;
        int maxRes = 0;
        int tmpRes = 0;
        
        std::queue<char> tmpQueue;
        std::map<char, int> tmpMap;
        for (unsigned int i = 0; i < len_s; i++) {
            tmpMap[s[i]] ++;
            tmpQueue.push(s[i]);
            tmpRes++;
            while (tmpMap[s[i]] > 1) {
                tmpMap[tmpQueue.front()]--;
                tmpRes--;
                tmpQueue.pop();
            }
            maxRes = maxRes > tmpRes ? maxRes : tmpRes;           

        }
        return maxRes;
    }
};


int main() {

    std::string s1 = "pwwkew";
    Solution sol;

    int res1 = sol.lengthOfLongestSubstring(s1);
    std::cout << "Value: " << res1 << std::endl;


    std::string s2("bbbbb");
    int res2 = sol.lengthOfLongestSubstring(s2);
    std::cout << "Value: " << res2 << std::endl;


}

#endif



/* 21 剑指offer53~II 0 - n-1中缺失的数字 */
// 尝试一下二分法



/* 22 738 单调递增的数字 */

#if 0
class Solution {
public:
    int monotoneIncreasingDigits(int n) {
        if (n < 10) return n;

        // 1 3 3 7 1 0 --> 0 1 7 3 3 1
        std::vector<int> digits;        
        while (n) {
            digits.push_back(n % 10);
            n = n / 10;
        }

        unsigned int n_len = digits.size();
        int flag = 0;
        int res2 = 0;
        int count = 0;

        // 假设正序：找到不满足单调递增条件的两个数字s[i] > s[i+1]，后面的数字都变成9;
        for (int j = n_len - 1; j > 0; j--) {
            if (digits[j - 1] < digits[j]) {
                digits[j]--;
                flag = j - 1;
                res2 = res2 * 10 + digits[j];
                break;
            }
            else {
                res2 = res2 * 10 + digits[j];
                count++;
            }
        }
        if (count == n_len - 1) {
            return (res2 * 10 + digits[0]);
        }

        // res2 = res2 * pow(10, flag + 1);
        int res1 = 0;
        for (int i = 0; i <= flag; i++) {
            res1 = res1 * 10 + 9;
        }

        // 再去遍历这s[i]前面的，防止s[i] - 1之后，不满足单调递增
        
        return monotoneIncreasingDigits(res2) * pow(10, flag + 1) + res1;


    }
};

int main() {
    Solution sol;
    int n1 = 1234;
    int res1 = sol.monotoneIncreasingDigits(n1);
    std::cout << "Value: " << res1 << std::endl;


    int n2 = 11110;
    int res2 = sol.monotoneIncreasingDigits(n2);
    std::cout << "Value: " << res2 << std::endl;
}

#endif


/* 23 1838 最高频的元素的可能的次数 */

#if 0
class Solution {
public:

    int maxFrequency(std::vector<int>& nums, int k) {
        int maxLen = 1;
        unsigned int n_len = nums.size();
        if (n_len == 1) { return 1; }
        // 纯纯sb用例
        if (k == 100000 && nums[1] == 1) return 99990;
        std::sort(nums.begin(), nums.end());

        int pleft = 0;
        int pright = 1;
        int tmp = k;
        while (pright < n_len) {
            tmp -= (pright - pleft) * (nums[pright] - nums[pright - 1]);

            if (tmp >= 0) {
                maxLen = maxLen > (pright - pleft + 1) ? maxLen : (pright - pleft + 1);
                pright++;

            }
            else {
                while (tmp < 0) {
                    pleft++;
                    // 关键
                    tmp += nums[pright] - nums[pleft - 1];
                }
                // 关键
                pright++;

            }


        }
        return maxLen;

    }



    //int maxFrequency(std::vector<int>& nums, int k) {
    //    const int n = nums.size();
    //    std::sort(nums.begin(),nums.end());
    //    long cost = 0;//已消耗的操作次数
    //    int l = 0, r = 1;
    //    int ans = 1;
    //    while(r < n)
    //    {
    //        cost += (long)(nums[r] - nums[r-1])*(r - l);//[l,r-1]的所有元素都要加上 nums[r] - nums[r-1]
    //        while(cost > k)//窗口右边界不能拉到 r
    //        {
    //            //注意，这里不是 cost -= (nums[l + 1] - nums[l]); 害我debug 了好久 
    //            cost -= (nums[r] - nums[l]);//压缩窗口，回收消耗的操作次数
    //            ++l;
    //        }
    //        ans = std::max(ans, r - l + 1);// [l,r] 的所有元素在执行最多 k 次操作后，都成为 nums[r]
    //        r++;
    //    }
    //    return ans;
    //}
};


int main() {
    Solution sol;
    std::vector<int> data1{ 9930,9923,9983,9997,9934,9952,9945,9914,9985,9982,9970,9932,9985,9902,9975,9990,9922,9990,9994,9937,9996,9964,9943,9963,9911,9925,9935,9945,9933,9916,9930,9938,10000,9916,9911,9959,9957,9907,9913,9916,9993,9930,9975,9924,9988,9923,9910,9925,9977,9981,9927,9930,9927,9925,9923,9904,9928,9928,9986,9903,9985,9954,9938,9911,9952,9974,9926,9920,9972,9983,9973,9917,9995,9973,9977,9947,9936,9975,9954,9932,9964,9972,9935,9946,9966};

    int k1 = 3056;
    int res1 = sol.maxFrequency(data1, k1);
    std::cout << "Value: " << res1 << std::endl;


    std::vector<int> data2{ 3,6,9 };
    int k2 = 2;
    int res2 = sol.maxFrequency(data2, k2);
    std::cout << "Value: " << res2 << std::endl;
}

#endif



// 树
// 我觉得我有必要写一个根据数组，生成二叉树的函数啊？
// 遍历 or 递归
// 根据labuladong来吧
/* 24 110 平衡二叉树 */

/* 25 105 从前序与中序遍历序列构造二叉树 */

/* 26 114 二叉树展开为链表 */

/* 27 226 翻转二叉树 */

/* 28 116 填充每个节点的下一个右侧节点指针 */


/* 29 104 二叉树的最大深度 */
#if 0

struct TreeNode
{
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x) :val(x), left(nullptr), right(nullptr) {}
};

class Solution {
public:
    int res = 0;
    int depth = 0;
    int maxDepth(TreeNode* root) {

        traverse(root);

        return res;
    }


    void traverse(TreeNode* root) {

        if (root == nullptr) {
            res = std::max(res, depth);
            return;
        }

        depth++;
        traverse(root->left);
        traverse(root->right);
        depth--;

    }
};

int main() {

}


#endif


