/*  
��ͷ��ʼˢleetcode
Ŀ�꣺���150+
*/

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>

// �����������
/* 1 695 ����������� */
// DFS �ݹ�ܹؼ�
// Ҫ����һЩȫ�ֱ���
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



/* 2 547 ʡ������ */
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




// ̰���㷨
/* 3  455  �ַ����� */
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



/* 4 206 ��ת���� */
// �ǵ�ѧ���ʼ������Ͷ���������
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


/* 5 135 �ַ��ǹ� */
#if 0

// n>=1
// points[i] >= 0
class Solution {
public:
    int candy(std::vector<int>& points) {
        
        // ��һ�֣�ȷ��ÿ�����Ӷ���1���ǹ���
        int n = points.size();

        if (n == 1) return 1;

        std::vector<int> nCandy(n, 1);


        // �ڶ��֣��������ң��ұ�>��ߣ���+1
        // ���Ҵ�ҳ�ʼ���ǹ���Ŀ����1
        for (int left = 1; left < n; left++) {
            if (points[left] > points[left - 1]) {
                nCandy[left] = nCandy[left - 1] + 1;
            }
        }


        // �ڶ��֣������������>�ұߣ���ȡmax(��ǰ�ģ��ұߵ�+1��
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



/* 6 435 �ص����� */
// �������
// �����Ժ���ô����
// flag�������
// ���޸�ֵ�ǵ�ʹ�� reference
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




/* 7 605 �ֻ����� */
// 1 <= flowerbed.length <= 2 * 10^4
// flowerbed[i] Ϊ 0 �� 1
// flowerbed �в��������ڵ����仨
// 0 <= n <= flowerbed.length

#if 0
class Solution {
public:
    bool canPlaceFlowers(std::vector<int>& flowerbed, int n) {
        int nflowerbed = flowerbed.size();
        int nexpend = nflowerbed + 2;

        std::vector<bool> flag(nexpend, false);

        // �����̳�Ǹ�λ�����˻�����ô��λ�ú�������λ�þͲ������ֻ���
        for (int i = 0; i < nflowerbed; i++) {
            if (flowerbed[i] == 1) {
                flag[i + 1] = true;
                flag[i] = true;
                flag[i + 2] = true;
            }
        }

        // ���㻨̳�����ֻ���λ����Ŀ
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


/* 8 452 �����ٵļ�ȥ�䱬���� */
// ����̰�����⣺����n��������[x,y]����������Ҫȷ�����ٸ��㣬ʹ��ÿ���������ж����ٴ���һ���㡣 (�����˵�ĺð�
#if 0

class Solution {
public:
    int findMinArrowShots(std::vector<std::vector<int>>& points) {
        
        int nBalloons = points.size();
        if (nBalloons == 1) return 1;

        // ���������У������ұ�С�ķ���ǰ��
        std::sort(points.begin(), points.end(), [](std::vector<int>& x, std::vector<int>& y) {
            return x[1] < y[1];
            });

        // ��Ϊ������Ŀ>=1, ����������Ҫһ����
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



/* 9 122 ������Ʊ�����ʱ�� */

#if 0

class Solution {
public:
    int maxProfit(std::vector<int>& prices) {
        int money = 0;
        int ndays = prices.size();
        if (ndays == 1) return 0;

        // ֻ��Ҫ�ж���i��͵�i+1���Ʊ�۸��С��
        // prices[i] < prices [i+1], ���������
        // else: ������
        // ʵ���ϴ�����Լ�������
        // ʵ�����ⲽֱ�ӷ����������ж������Ϳ�����
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


// ˫ָ��
/* 10 1 ����֮�� */
// ��Ŀ�нϴ���޸�Ŷ����������ô�򵥵�����������
// �����������ݸо�������
#if 0
class Solution {
public:
    std::vector<int> twoSum(std::vector<int>& nums, int target) {
        std::vector<int> backNums = nums;

        std::sort(nums.begin(), nums.end(), [](int a, int b) {return a < b; });

        int left = 0;
        int right = nums.size() - 1;

        // ��ȡ��ʵ�ʵ�ֵ nums[left] and nums[right]
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

        // ����ֵȥ����������

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


/* 11 167 ����֮��--�������� */
// ����������ռ�
#if 1
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

/* 12 88 �鲢������������ */


