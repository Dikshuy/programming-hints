## Hints and implemenation of some leetcode and hackerrank problems

### Longest Increasing Path in a matrix
Issue: Optimal way of searching using DFS(Depth First Search algorithm). [Problem link](https://leetcode.com/problems/longest-increasing-path-in-a-matrix/)

Hint: Refer this [video](https://www.youtube.com/watch?v=7fujbpJ0LB4&ab_channel=WilliamFiset) for understanding DFS.

> Approach: Depth-first search is an algorithm for traversing or searching tree or graph data structures. The algorithm starts at the root node (selecting some arbitrary node as the root node in the case of a graph) and explores as far as possible along each branch before backtracking. So the basic idea is to start from the root or any arbitrary node and mark the node and move to the adjacent unmarked node and continue this loop until there is no unmarked adjacent node. Then backtrack and check for other unmarked nodes and traverse them.

**Implementation**
```python
class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        if not matrix:
            return 0
        rows = len(matrix)
        cols = len(matrix[0])
        traced = {}
        ans = 0
        for i in range(rows):
            for j in range(cols):
                path = self.search(matrix, i, j, traced) 
                ans = max(ans, path)
        return ans
    
    def search(self, matrix, i, j, traced):
        rows = len(matrix)
        cols = len(matrix[0])
        
        if (i,j) in traced:
            return traced.get((i,j))
        
        dirs = [(-1,0),(1,0),(0,1),(0,-1)]        
        path = 1
        
        for x, y in dirs:
            new_x = x+i
            new_y = y+j
            if rows>new_x>=0 and cols>new_y>=0 and matrix[new_x][new_y]>matrix[i][j]:
                path = max(path, 1+self.search(matrix, new_x, new_y, traced))
        traced[(i,j)] = path           
        return traced[(i,j)]
```

### Longest Increasing Subsequence
Issue: Find out an optimal solution using dynamic programming. [Problem link](https://leetcode.com/explore/challenge/card/july-leetcoding-challenge-2021/609/week-2-july-8th-july-14th/3808/) 

Hint: Use memoization or tabulation. [video](https://youtu.be/4fQJGoeW5VE) for reference

**Implementation**
```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        L = [1]*len(nums)
        for i in range(1, len(nums)):
            for j in range(i):
                if nums[j] < nums[i] and L[i] < L[j]+1:
                    L[i] = L[j]+1
        max_ = 0
        for i in range(len(nums)):
            max_ = max(max_, L[i])
        return max_
```
Future work: Above implementation takes `O(N*N)` time complexity, reduce it `O(NlogN)`. Refer this [link](https://www.geeksforgeeks.org/construction-of-longest-monotonically-increasing-subsequence-n-log-n/)

### Decode Ways
Issue: To decode an encoded message, all the digits must be grouped then mapped back into letters using the reverse of the mapping above. [Problem link](https://leetcode.com/problems/decode-ways/)

Hint: For recursive solution, take either only the first or first two digits of the given number and recurse through the rest in a similar manner.

**Implementation**

**recursive**
```python
class Solution:
    def numDecodings(self, s: str) -> int:
        if s[0] == "0": return 0
        return self.sub(s)
    def sub(self,string):
        if not string:
            return 1
        first = second = 0
        if 1 <= int(string[:1]) <= 9:
            first = self.sub(string[1:])
        if 10 <= int(string[:2]) <= 26:
            second = self.sub(string[2:])
        return first+second             
```
**dynamic programming**
```python
class Solution:
    def numDecodings(self, s: str) -> int:             
        if s[0] == "0": return 0         
        dp = [1] * (len(s) + 1)

        for i in range(2, len(s) + 1):
            dp[i] = (dp[i - 1] if 1 <= int(s[i - 1]) <= 9 else 0) + (dp[i - 2] if 10 <= int(s[i - 2] + s[i - 1]) <= 26 else 0)
        
        return dp[-1]               
```

### Russian Doll envelops
Issue: Minimize time complexity and solve using DP and binary search. [Problem link](https://leetcode.com/problems/russian-doll-envelopes/)

Hint: Use the logic of LIS. Also, go through this [bisect](https://www.geeksforgeeks.org/bisect-algorithm-functions-in-python/) library which is used to find a position in list where an element needs to be inserted to keep the list sorted

**Implementation**
```python
class Solution:
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        # For each envelope, sorted by envelope[0] first, so envelope[1] is the the longest
        # increasing sequence(LIS) problem. When envelope[0] tie, we reverse sort by envelope[1]
        # because bigger envelope[1] can't contain the previous one.
        envelopes.sort(key=lambda e: (e[0], -e[1]))
        # dp keeps some of the visited element in a sorted list, and its size is length Of LIS
        # so far. It always keeps the our best chance to build a LIS in the future.
        dp = []
        for envelope in envelopes:
            i = bisect.bisect_left(dp, envelope[1])
            if i == len(dp):
                # If envelope[1] is the biggest, we should add it into the end of dp.
                dp.append(envelope[1])
            else:
                # If envelope[1] is not the biggest, we should keep it in dp and replace the
                # previous envelope[1] in this position. Because even if envelope[1] can't build
                # longer LIS directly, it can help build a smaller dp, and we will have the best
                # chance to build a LIS in the future. All elements before this position will be
                # the best(smallest) LIS sor far. 
                dp[i] = envelope[1]
        # dp doesn't keep LIS, and only keep the length Of LIS.
        return len(dp)
```

### Sort the matrix Diagonal
Issue: [link](https://leetcode.com/problems/sort-the-matrix-diagonally/submissions/)

Hint:Store the matrices diagonal in `collections.defaultdict(list)` and sort them

**Implementation**
```python
class Solution:
    def diagonalSort(self, mat: List[List[int]]) -> List[List[int]]:
        dict=collections.defaultdict(list)
        n=len(mat)
        m=len(mat[0])
        for i in range(0,n):
            for j in range(0,m):
                dict[n-1-i+j].append(mat[i][j])
    
        for i in dict:
            dict[i].sort()
            
        ans=[[0 for i in range(0,m)] for i in range(0,n)]
        for i in range(0,n):
            for j in range(0,m):
                print(dict[n-1-i+j])
                ans[i][j]=dict[n-1-i+j][0]
                dict[n-1-i+j].pop(0)
                
        return ans
```
### Merge k Sorted Linked Lists
Issue: linked lists and have to return as a linked list. [link](https://leetcode.com/problems/merge-k-sorted-lists/)

Hint: Decode and encode linked list

**Implementation**
```python
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        
        def helper(node):
            nodes = []
            while node: 
                nodes.append(node)
                node = node.next
            return nodes
        
        nodes = []
        for node in lists:
            nodes.extend(helper(node))
        
        if not nodes:
            return 
        
        # print(nodes)
        # print(type(nodes))
        nodes.sort(key = lambda x: x.val)
        
        for node1, node2 in zip(nodes, nodes[1:]):
            node1.next = node2
        
        nodes[-1].next = None
        
        return nodes[0]
```

### Maximum length of repeated subarray
Issue: reduce time complexity using DP. [link](https://leetcode.com/explore/challenge/card/july-leetcoding-challenge-2021/609/week-2-july-8th-july-14th/3807/)

Hint: Maintain a new 2d array of zeros and update whenever you sth common.

**Implementation**
```python
class Solution:
    def findLength(self, nums1: List[int], nums2: List[int]) -> int:
        dp = [[0 for i in range(len(nums1) + 1)] for i in range(len(nums2) + 1)]
        for i in range(len(nums1)-1,-1,-1):
            for j in range(len(nums2)-1,-1,-1):
                if nums1[i] == nums2[j]:
                    dp[j][i] = dp[j+1][i+1] + 1
        num = 0
        for i in dp:
            for j in i:
                num = max(num,j)
        return num
```

### Adding Two numbers
Issue: Numbers are stored in linked lists, so how to access them properly. [link](https://leetcode.com/problems/add-two-numbers/)

Hint: Convert nodes to a list and then back to linked list.

**Implementation**
```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def node_to_list(self, listnode):
        l=[]
        while True:
            l.append(listnode.val)
            if listnode.next != None:
                listnode = listnode.next
            else:
                return l
            
    def list_to_LL(self,arr):
        if len(arr) < 1:
            return None

        if len(arr) == 1:
            return ListNode(arr[0])
        return ListNode(arr[0], next=self.list_to_LL(arr[1:]))

#     def reverseList(head: ListNode) -> ListNode:
#         prev = None
#         while head:
#             next_node = head.next
#             head.next = prev
#             prev = head
#             head = next_node

#         return prev
    
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        
        l1 = self.node_to_list(l1)
        l2 = self.node_to_list(l2)
        num1=0
        num2=0
        for i in range(len(l1)):
            num1+=l1[i]*(10**i)
        for i in range(len(l2)):
            num2+=l2[i]*(10**i)
        num = num1+num2
        l = []
        if num==0:
            l=[0]
        while num>0:
            l.append(num%10)
            num=num//10
            
        return self.list_to_LL(l)
```

### Larry's array:
Issue of the problem: Test whether array can be sorted by swaping three items at a time in order: ABC -> BCA -> CAB -> ABC. [Link](https://www.hackerrank.com/challenges/larrys-array/problem?isFullScreen=true)

Hint: Check out the number of inversions. The given below implementation seems to be simple but there is an awesome logic behind this. Refer to these paper's for understanding the logic behind it. [Paper1](https://www.cs.bham.ac.uk/~mdr/teaching/modules04/java2/TilesSolvability.html) and [Paper2](http://kevingong.com/Math/SixteenPuzzle.html)

**Implementation**:
```python
string larrysArray(vector<int> A) {
    int n=A.size();
    int sum=0;
    for(int i = 0; i < n; i++){
        for(int j = i+1; j < n; j++){
            if(A[j] < A[i]){
                sum += 1;
            } 
        }
    }
    if(sum%2==0){
        return "YES";
    }
    else{
        return "NO";
    }
} 
```
### Minimum Loss:
Issue of the problem: Time complexity issue in case of larger values. [link](https://www.hackerrank.com/challenges/minimum-loss/problem)

Hint: Sort the array, and then check the difference of adjacent pairs, if its less than ur last min value, update it only if the index of those pairs are in same way in original array.

**Implementation**:
```python
    int n;
    cin >> n;
    vector<double> sorted(n);
    map<double, int> arr;
    for (int i = 0; i < n; ++i) {
        double x;
        cin >> x;
        sorted[i] = x;
        arr[x] = i;
    }
    sort(sorted.begin(), sorted.end());
    double min = INT_MAX;
    for (int i = 0; i < n - 1; ++i) {
        double x = sorted[i + 1] - sorted[i];
        if (x < min) {
            int first = arr[sorted[i]];
            int second = arr[sorted[i + 1]];
            if (second < first) {
                min = x;
            }
        }
    }
    cout << long(min);
```
### Power Sum
Issue of the problem: [link](https://www.hackerrank.com/challenges/the-power-sum/problem)

Hint: (This hint I found in the discussion panel and is a very easy implementation of recursion).

* At any point, either we can either use that number or not.
* If we do not use it then X value will remain same.
* And if we use it, then we have to subtract pow(num, N) from X.
* num value will increase every time as we can use one number at most once.
* Our answer will be sum of both these cases. This is obvious.
* And then we will do same thing for two values of X i.e. X and X-pow(num,N).
* If value of X is less than pow(num, N) then we cannot get answer as value of num will keep increasing. Hence, we return 0.
* If it is equal to 1, then we can return 1.

**Implementation**:
```python
int powerSum(int X,int N,int num){
    if(pow(num,N)<X)
        return powerSum(X,N,num+1)+powerSum(X-pow(num,N),N,num+1);
    else if(pow(num,N)==X)
        return 1;
    else
        return 0;
}
```

### Factorial of a large number:
Issue of the problem: Large factorials can't be stored even in case of long long int. So, the given below is a idea for solving such cases.

Hint: Initialize a matrix of a large size ,let's say, 1000. Put its start value as 1 and one other parameter size as 1. Now, as you peform normal multplication update the values.

**Implementation**:
```python
void extraLongFactorials(int n) {
    int val[1000];
    int size = 1;
    val[0] = 1;
    size = 1;
    for(int i=2; i<=n; i++){
        int carry = 0;
        for(int j=0; j<size; j++){
            int pod = val[j]*i+carry;
            val[j] = pod%10;
            carry = pod/10;
        }
        while(carry){
            val[size] = carry%10;
            carry/=10;
            size++;
        }
    }
    for(int i = size-1; i>= 0; i--)cout << val[i];
}
```

### Queen's attack:
Problem: Given the queen's position and the locations of all the obstacles, find and print the number of squares the queen can attack from her position at (r_q, c_q).

Hint: Initialize the distances from the current position to the end of the chessboard in every direction to its actual distance. Then check along every direction and when any obstacle comes in front, set that distance as the value along that direction.

**Implemenatation**:
```python
int queensAttack(int n, int k, int r_q, int c_q, vector<vector<int>> obstacles) {

queen_row = r_q
queen_column = c_q

top = n - queen_row
bottom = queen_row - 1
right = n - queen_column
left = queen_column - 1

top_left = min(n - queen_row, queen_column - 1)
top_right = n - max(queen_column, queen_row)
bottom_left = min(queen_row, queen_column) - 1
bottom_right = min(queen_row - 1, n - queen_column)

for a0 in xrange(k):
    obstacle_row = obstacles[a0][0]
    obstacle_column = obstacles[a0][1]

    if obstacle_row == queen_row:
        if obstacle_column > queen_column:
            top = min(top, obstacle_column - queen_column - 1)
        else:
            bottom = min(bottom, queen_column - obstacle_column - 1)
 
    elif obstacle_column == queen_column:
        if obstacle_row > queen_row:
            right = min(right, obstacle_row - queen_row - 1)
        else:
            left = min(left, queen_row - obstacle_row - 1)

    elif abs(obstacle_column - queen_column) == abs(obstacle_row - queen_row):
   
        if obstacle_column > queen_column and obstacle_row > queen_row:
            top_right = min(top_right, obstacle_column - queen_column - 1)
      
        elif obstacle_column > queen_column and obstacle_row < queen_row:
            bottom_right = min(bottom_right, obstacle_column - queen_column - 1)
        
        elif obstacle_column < queen_column and obstacle_row > queen_row:
            top_left = min(top_left, queen_column - obstacle_column - 1)
       
        elif obstacle_column < queen_column and obstacle_row < queen_row:
            bottom_left = min(bottom_left, queen_column - obstacle_column - 1)
    
            
print top + bottom + right + left + top_left + top_right + bottom_left + bottom_right 
}
```

### Oraginizing Containers of balls:
Issue of the problem: [link](https://www.hackerrank.com/challenges/organizing-containers-of-balls/problem?isFullScreen=false)

Hint: 1. Make a vector of capacity of every box  2. Make a vector of all the balls  3. Sort both of them 

Compare both the vectors. If same then possible else impossible.

**Implementation**:
```python
string organizingContainers(vector<vector<int>> container){
    vector<int> capacity;
    vector<int> balls;
    for(unsigned int i=0; i<container.size(); i++){
        int cols = 0;
        int rows = 0;
        for(unsigned int j=0; j<container.size(); j++){
            cols += container[i][j];
            rows += container[j][i];
        }
        balls.push_back(cols);
        capacity.push_back(rows);
    }   
    sort(balls.begin(), balls.end());
    sort(capacity.begin(), capacity.end());
    
    if(balls == capacity){
        return "Possible";
    }
    else{
        return "Impossible";
    }

}
```
### Almost sorted:
Issue of the problem: Checking whether a vector can be sorted using reverse or swap operation. [link](https://www.hackerrank.com/challenges/almost-sorted/problem?isFullScreen=true)

Hint: * Run through the vector from index 1 to len-2 ( leaving the first and last elements)

* At each of these indices check whether it forms an inversion or a reverse inversion. Inversion is if curr > prev && curr > next. Similarly find out reverse inversions, curr < prev && curr < next. I call inversions as dips, and reverse inversions as ups. For the first and last elements you can check only the next and prev respectively as they are at the boundary.

* Once you have collected data of these inversions, if you analyze you will see that if reverse has to form a soln, you will have only one dip and one up.

* And if swapping can be soln then there will be 2 dips and 2 ups.

* If you get more than 2 dips and 2ups, it means it can't be solved.

* There are some edge cases which you need to take care of though.

A relevant you tube [video](https://www.youtube.com/watch?v=UWmSQFNjEZg&feature=youtu.be) to get a deeper insight of above algorithm.

### 3D surface area:
Issue of the problem: [link](https://www.hackerrank.com/challenges/3d-surface-area/problem?isFullScreen=true)

Hint: The base of the Figure will always contribute to the total surface area of the figure. Now, to calculate the area contributed by the walls, we will take out the absolute difference between the height of two adjacent wall. The difference will be the contribution in the total surface area.

**Implementation**
```python
int contribution_height(int current, int previous) { 
    return abs(current - previous); 
}
int surfaceArea(vector<vector<int>> A) {
    int ans = 0; 
    int N = A.size();
    int M = A[0].size();
     
    for (int i = 0; i < N; i++) { 
        for (int j = 0; j < M; j++) {
            int up = 0; 
            int left = 0; 
  
            if (i > 0) 
                up = A[i - 1][j]; 
  
            if (j > 0) 
                left = A[i][j - 1]; 
   
            ans += contribution_height(A[i][j], up)  
                    + contribution_height(A[i][j], left); 
 
            if (i == N - 1) 
                ans += A[i][j]; 
            if (j == M - 1) 
                ans += A[i][j]; 
        } 
    } 
  
    // Adding the contribution by the base and top of the figure 
    ans += N * M * 2; 
    return ans;
}
```

### Absolute Permutation
Issue: Represents the smallest lexicographically smallest permutation of natural numbers, such that |pos[i]-i|=k.  [link](https://www.hackerrank.com/challenges/absolute-permutation/problem?isFullScreen=true)

Hint: Distribute into k and swap between 2k.

**Implementation**:
```python
vector<int> absolutePermutation(int n, int k) {
    vector<int> pos(n);
    for(int i=0; i<n; i++){
        pos[i]=i+1;
    }
    vector<int> permutation(n);
    if(k!=0){
        if(n%k!=0 || (n/k)%2!=0 || k>n/2){
        return {-1};
        }
        for(int m=0; m<n; m=m+2*k){
            for(int j=0;j<k;j++){
                swap(pos[m+j], pos[m+j+k]);
            }
        }
    }
    permutation = pos;

    return permutation;
}
```

### Ordering the team
Issue: Check whether the teams can be ranked on the basis of three parameters. [link](https://www.hackerrank.com/contests/pt-test-3/challenges/ordering-the-team/problem)

Hint: Make a 2d vector and sort it. Then do comparisons in rows and change the bool if not possible.

**Implementation**:
```python
int n;
    cin>>n;
    vector<vector<int>> v(n);
    for (int i=0;i<n;i++)
    {
        int a, b, c;
        cin>>a>>b>>c;
        v[i].push_back(a);
        v[i].push_back(b);
        v[i].push_back(c);
    }
    sort(v.begin(), v.end());
  
    bool ans = true;
    for (int i=0;i<n-1;i++){    
        int count = 0;
        for (int j=0;j<3;j++){
            if (v[i][j] < v[i+1][j])
                count++;
            else if (v[i][j] > v[i+1][j])
                ans = false;
        }
        if (count == 0)
            ans = false;
    }
    if (ans)
        cout<<"Yes";
    else
        cout<<"No";
```

### Reduce array size to half
Issue: sort the dictionary by values. [link](https://leetcode.com/explore/challenge/card/july-leetcoding-challenge-2021/608/week-1-july-1st-july-7th/3804/)

Hint: Use counter function from collections library and then `Counter(arr).most_common` to sort the counter dicitonary according to the values or can use this: `sorted(Counter(arr).items(), key=lambda x: x[1], reverse=True)`
