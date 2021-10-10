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

### Making a large Island
Issue: this current implementation is using runtine complexity as O(N^4), try to optimize it. [Problem link](https://leetcode.com/problems/making-a-large-island/)

Hint: For each 0, change it to a 1, then do a depth first search to find the size of that component. The answer is the maximum size component found.

**Implementation**
```python
class Solution:
    def largestIsland(self, grid: List[List[int]]) -> int:
        
        def search(i,j):
            seen = {(i,j)}
            stack = [(i,j)]
            while stack:
                i,j = stack.pop()
                for new_i,new_j in ((i-1,j),(i+1,j),(i,j-1),(i,j+1)):
                    if 0<=new_i<len(grid) and 0<=new_j<len(grid[0]) and (new_i,new_j) not in seen and grid[new_i][new_j]:
                        stack.append((new_i, new_j))
                        seen.add((new_i, new_j))
            return len(seen)
       
        has_zero = False
        ans = 0
        for i in range(0, len(grid)):
            for j in range(0, len(grid[0])):
                if grid[i][j]==0:
                    has_zero = True
                    grid[i][j]=1
                    ans = max(ans, search(i,j))
                    grid[i][j]=0
                    
        return ans if has_zero else len(grid)*len(grid[0])
```

### Dungeon Game
Issue: dynamic programming [problem link](https://leetcode.com/problems/dungeon-game/)

Hint: revert back from the final position
> Basically, this problem would work from front to back, but the optimal solution for traversing from (0, 0) to (i, j) will not always give us the optimal solution for (i + 1, j) and (i, j + 1). Sometimes it is better to take the locally worse route to be able to have enough HP so that the negatives that will be encountered later are minimized.

However, if we work in reverse, we can avoid this. Start from (n - 1, m - 1), where n is the number of rows, and m is the number of columns.
In order to make the space complexity O(N), we use the concept of time iterations.

at t=0, the Knight is at (0, 0)

at t=1, the Knight is at (1,0) or (0,1)

at t=2, the Knight is at (2,0), (1,1) or (0,2)

So, we can sort of make the relation between t and an arbitrary point (i, j) that the knight may be:
t = i + j.
Therefore, we can iterate through all of the t values, and store them in a dp array of size n.

> Note that this dp only stores values of the rows. that's because if we know the row and the current time, then we know the column j = t - i.

So, we can see that there are n + m - 1 iterations of t: [0, n + m - 2], and in order to start out the dp, we go ahead and calculate dp[n-1], which corresponds to t = n + m - 2, and dungeon[n - 1][m - 1]

Here's an example runthrough of the logic:

            [[-2,-3,3],
            [-5,-10,1],
            [10,30,-5]]
			Initialize dp with dungeon[n - 1][m - 1]
            t = 4 i = 2, j = 2,  the -5 means we need 1 - (-5) HP = 6HP.... dp = [inf, inf, 6]
            
            t = 3 i = 2, j = 1   the 30 > 6, so we dont need 6 minHP anymore, minHP = 1
            t = 3 i = 1, j = 3   the 1 < 6, so we need 1 less HP, so minHP = 6 - 1 = 5
			...dp = [inf, 5, 1]
            
            t = 2 i = 2 j = 0.   the 10 > 1, so we still need 1
            t = 2 i = 1 j = 1.   the -10 < 1 and -10 < 5 minHP = min(1, 5) - (-10) = 11
			t = 2 i = 0 j = 2.   the  3 < 5 so minHP = 5 - 3 = 2
			...dp = [2,11,1]
			
			t = 1 i = 1 j = 0    the -5 < 11, and -5 < 1  so minHP = min(1, 11) - (-5) = 6
			t = 1 i = 0 j = 1    the -3 < 2 and -3 < 11 so minHP = min(2, 11) - (-3) = 5
			...dp = [5, 6, inf]
			
			t = 0 i = 0 j = 0.   the -2 < 5 and -2 < 6 so minHP = min(5, 6) -(-2) = 7HP
			...dp = [7,inf,inf]
			
			then return dp[0] = 7

**Implementation**
```python
class Solution:
    def calculateMinimumHP(self, dungeon: List[List[int]]) -> int:
        n = len(dungeon)
        m = len(dungeon[0])
        dp = [float('inf') for _ in range(len(dungeon))]
        if dungeon[n-1][m-1] >= 0:
            dp [n-1] = 1
        else:
            dp[n-1] = 1 - dungeon[n-1][m-1]
        for t in range(n + m - 3, -1, -1):
            dp2 = [float('inf') for _ in range(n)]
            for i in range(max(0, t - m + 1), min(n, t + 1)):
                j = t - i
                if i + 1 < n:
                    if dungeon[i][j] < dp[i + 1]:
                        dp2[i] = dp[i + 1] - dungeon[i][j]
                    else:
                        dp2[i] = 1
                if j + 1 < m:
                    if dungeon[i][j] < dp[i]:
                        dp2[i] = min(dp2[i], dp[i] - dungeon[i][j])
                    else:
                        dp2[i] = 1
            dp = dp2
        return dp[0]
```

### Largest Rectangle in Histrogram
Issue: [problem link](https://leetcode.com/problems/largest-rectangle-in-histogram/)

Hint: Think in lines of recursion

**Implementation**
```python
# getting wrong solution currently
```

### Word Search  II
Issue: search a lists of words in a grid. [problem link](https://leetcode.com/problems/word-search-ii/)

Hint: use DFS similar to problem #1

**Implementation**
```python
class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        if not board or not words: return False
        traced = set()
        
        
        def dfs(start, i, j, traced):
            dirs = [(0,1), (1,0), (-1,0), (0,-1)]
            print(start)
            for dx, dy in dirs:
                new_x = i+dx
                new_y = j+dy
                if 0 <= new_x < len(board) and 0 <= new_y < len(board[0])  and (new_x,new_y) not in traced and board[new_x][new_y] == word[start]:
                    if start == len(word)-1:    return True

                    traced.add((new_x, new_y))
                    if dfs(start+1, new_x, new_y, traced):  return True
                    else:   traced.remove((new_x, new_y))
            return
        
        ls = []
        for word in words:
            start = 0
            for i in range(len(board)):
                for j in range(len(board[0])):
                    traced = set()
                    if board[i][j] == word[start]:			
                        traced.add((i,j))
                        if start == len(word) -1:
                            ls.append(word)
                            continue
                        if dfs(start+1, i, j, traced): 
                            ls.append(word)
        
        final = []
        for i in ls:
            if i not in final:
                final.append(i)
        
        return final  
	
# need to improve code, exceeding time limit for few cases (look out for TRIE)
```

### Castle on the grid
Issue: [problem link](https://www.hackerrank.com/challenges/castle-on-the-grid/problem)

Hint: Use BFS

**Implementation**
```python
def minimumMoves(grid, startX, startY, goalX, goalY):
    if not grid: return 0
    queue = deque()
    rows = len(grid)
    cols = len(grid[0])
    directions=[(1,0),(-1,0),(0,-1),(0,1)]
    queue.appendleft((startX,startY,0))
    visited = set()
    while queue:
        (i,j,dist) = queue.pop()
        new_dist = dist + 1
        for d in directions:
            new_i = i + d[0]
            new_j = j + d[1]
            while 0 <= new_i < rows and 0 <= new_j < cols and grid[new_i][new_j]!='X':
                if (new_i, new_j) == (goalX, goalY):
                    return new_dist
                elif (new_i, new_j) not in visited:
                    queue.appendleft((new_i,new_j,new_dist))
                    visited.add((new_i,new_j))
                new_i += d[0]
                new_j += d[1]
```

### City of Blinding nights
Issue: [Problem link](https://www.hackerrank.com/challenges/floyd-city-of-blinding-lights/problem)

Hint: Bellman-Ford algorithm

**Implementation**
```python
class Graph:
    def __init__(self, vertices):
        self.V = vertices 
        self.graph = []
 
    def addEdge(self, u, v, w):
        self.graph.append([u, v, w])
        
    def shortest_distance(self,src,dest):
        dist = [float("Inf")] * self.V
        dist[src] = 0

        for _ in range(self.V - 1):
            for u, v, w in self.graph:
                if dist[u] != float("Inf") and dist[u] + w < dist[v]:
                        dist[v] = dist[u] + w
        if dist[dest] == float("Inf"): print(-1)
        else: print(dist[dest])
        
    
if __name__ == '__main__':
    road_nodes, road_edges = map(int, input().rstrip().split())
    g = Graph(road_edges)
    road_from = [0] * road_edges
    road_to = [0] * road_edges
    road_weight = [0] * road_edges

    for i in range(road_edges):
        road_from[i], road_to[i], road_weight[i] = map(int, input().rstrip().split())
        g.addEdge(road_from[i], road_to[i], road_weight[i])
    q = int(input().strip())

    for q_itr in range(q):
        first_multiple_input = input().rstrip().split()

        x = int(first_multiple_input[0])

        y = int(first_multiple_input[1])
        
        g.shortest_distance(x,y)
```

### Binary Tree Maximum Path Sum
Issue: [Problem link](https://leetcode.com/problems/binary-tree-maximum-path-sum/)

Hint: Use DFS 

**Implementation**
```python
class Solution:
    def maxPathSum(self, root: TreeNode) -> int:
        self.ans = float(-inf)
        def dfs(root):
            if not root: return 0
            self.ans = max(self.ans, dfs(root.left)+dfs(root.right)+root.val)  
            return max(0, root.val+max(dfs(root.left),dfs(root.right)))      # check whether the left or part returns max sum   
        dfs(root)
        return self.ans
```

### Shortest Path in  a grid with obstacle elimination
Issue: Removing obstacles. [Problem Link](https://leetcode.com/problems/shortest-path-in-a-grid-with-obstacles-elimination/)

Hint: Use Breadth First Search algorithm. [video](https://www.youtube.com/watch?v=oDqjPvD54Ss&ab_channel=WilliamFiset) for reference

**Implementation**
```python
class Solution:
    def shortestPath(self, grid: List[List[int]], k: int) -> int:
        q=deque()
        m=len(grid)
        n=len(grid[0])
        directions=[(1,0),(-1,0),(0,-1),(0,1)]
        q.append((0,0,k))
        visited=set()
        visited.add((0,0,k))
        ans=0
        while q:
            for v in range(len(q)):
                i,j,limit=q.popleft()
                if i==m-1 and j==n-1:
                    return ans
                for d in directions:
                    new_i=i+d[0]
                    new_j=j+d[1]
                    if 0<=new_i<m and 0<=new_j<n:
                        if grid[new_i][new_j]==0 and (new_i,new_j,limit) not in visited:
                            q.append((new_i,new_j,limit))
                            visited.add((new_i,new_j,limit))
                        elif limit>0 and (new_i,new_j,limit-1) not in visited:
                            q.append((new_i,new_j,limit-1))
                            visited.add((new_i,new_j,limit-1))
            ans+=1
        return -1
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

### Longest Common subsequence
Issue: Solve it efficiently using dynammic programming in O(mn) complexity. [Problem link](https://leetcode.com/problems/longest-common-subsequence/)

Hint: Refer to this [blog](https://www.programiz.com/dsa/longest-common-subsequence) or this [video](https://www.youtube.com/watch?v=LAKWWDX3sGw&t=989s)

**Implementation**
```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m = len(text1)
        n = len(text2)
        dp = [[0 for x in range(n+1)] for x in range(m+1)]
        for i in range(m+1):
            for j in range(n+1):
                if i == 0 or j == 0:
                    dp[i][j] = 0
                elif text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i][j-1], dp[i-1][j])
       
        return dp[-1][-1]
```

❗❗❗
### Sorted Subsegemnts
Issue: [Problem link](https://www.hackerrank.com/challenges/sorted-subsegments/problem)

Hint: no clue how to pass all the test cases

**Implemenation**
```python
def sortedSubsegments(k, a, queries):
    # Write your code here
    for i,j in queries:
        a = a[:i]+sorted(a[i:j+1])+a[j+1:]
    
    return a[k]

# this dumb solution can only pass 10 cases, think of some better approach
```

### Red Knight's Shortest Path
Issue: How to take different steps in search based on priority order. [Problem link](https://www.hackerrank.com/challenges/red-knights-shortest-path/problem)

Hint: no idea currently, need to solve it :(

**Implementation**
```python

```

### Array Manipulation
Issue: [Problem link](https://www.hackerrank.com/challenges/crush/problem)

Hint: Think in lines of [prefix sum](https://www.geeksforgeeks.org/prefix-sum-array-implementation-applications-competitive-programming/)

**Implementation**
```python
def arrayManipulation(n, queries):
    # Write your code here
    arr = [0]*n
    for i in range(len(queries)):
        arr = arr[:(queries[i][0]-1)] + [sum(x) for x in zip([queries[i][2]]*(queries[i][1]+1-queries[i][0]), arr[queries[i][0]-1:queries[i][1]])] + arr[queries[i][1]:]
   
    return max(arr)
    
# this solution is not optimal and runtime is exceeding, a better approach can be followed by using the concept of prefix sum
  
def arrayManipulation(n, queries):
    # Write your code here
    arr = [0]*(n+2)
    for i,j,k in queries:
        arr[i] += k
        arr[j+1] -= k
    max_ = temp = 0
    for val in arr:
        temp += val
        max_ = max(max_, temp)
    
    return max_
```

### Merge Intervals
Issue: [Problem link](https://leetcode.com/problems/merge-intervals/)

Hint: sort and store the overlapping intervals

**Implementation**
```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals=sorted(intervals)
        output=intervals[0]
    
        res=[]
        for i in range(1,len(intervals)):
            if intervals[i][0]<=output[1]:
                output[1]=max(output[1],intervals[i][1])         
            else:
                res.append(output.copy())
                output[0]=intervals[i][0]
                output[1]=intervals[i][1]
            
        res.append(output)
        return res
```

### Anagrams
Issue: [Problem link](https://www.hackerrank.com/challenges/sherlock-and-anagrams/problem)

Hint: create counter dictionary and anagrams pairs can be found by sorting the substring

**Implementation**
```python
def sherlockAndAnagrams(s):
    dictionary= {}
    for i in range(len(s)):
        for j in range(i,len(s)):
            substr  =''.join(sorted(s[i:j+1]))
            dictionary.setdefault(substr, 0)
            dictionary[substr] += 1
    count = 0
    for string in dictionary:
        count += sum([i for i in range(dictionary[string])])
    return count
```

### Text Justification
Issue: Formatting the chosen letters. [Problem Link](https://leetcode.com/problems/text-justification/)

Hint: Think of 1. How many words we need to form each line.
      2. How many spaces we should insert between two words.
      
**Implementation**
```python
class Solution(object):
    def fullJustify(self, words, maxWidth):
        '''
        :type words: List[str]
        :type maxWidth: int
        :rtype: List[str]
        '''
        n = len(words)
        L = maxWidth
        i = 0     # the index of the current word   
        ans = [] 
        
        def getKwords(i):
            k = 0 # figure out how many words can fit into a line
            l = ' '.join(words[i:i+k]) 
            while len(l) <= L and i+k <= n:
                k += 1
                l = ' '.join(words[i:i+k])
            k -= 1 
            return k
        
        
        def insertSpace(i, k):
            ''' concatenate words[i:i+k] into one line'''
            l = ' '.join(words[i:i+k])       
            if k == 1 or i + k == n:        # if the line contains only one word or it is the last line  
                spaces = L - len(l)         # we just need to left assigned it
                line = l + ' ' * spaces 
            else:                           
                spaces = L - len(l) + (k-1) # total number of spaces we need insert  
                space = spaces // (k-1)     # average number of spaces we should insert between two words
                left = spaces % (k-1)       # number of 'left' words, i.e. words that have 1 more space than the other words on the right side
                if left > 0:
                    line = ( " " * (space + 1) ).join(words[i:i+left])  # left words
                    line += " " * (space + 1)                           # spaces between left words & right words
                    line += (" " * space).join(words[i+left:i+k])       # right woreds
                else: 
                    line = (" " * space).join(words[i:i+k])
            return line
        

        while i < n: 
            k = getKwords(i)  
            line = insertSpace(i, k) # create a line which contains words from words[i] to words[i+k-1]
            ans.append(line) 
            i += k 
        return ans
```

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

### Merge in between linked list
Issue: Converting back to list and then retracing back exceeded the time limit. [link](https://leetcode.com/problems/merge-in-between-linked-lists/)

Hint: Just check the head and node.next of where we want to add the LL

**Implementation**
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeInBetween(self, list1: ListNode, a: int, b: int, list2: ListNode) -> ListNode:
        prev = head = ListNode(0, list1) # Sentinal node for edge a=1
        for _ in range(a): # Reach node before left
            head = head.next
        temp = head
        for _ in range(b - a + 1): # Reach right node
            head = head.next
        temp.next = list2 # Add newlist at left
        while list2.next: # Traverse the new list
            list2 = list2.next
        list2.next = head.next # Add nodes after right node
        return prev.next
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

### 4 Sum
Issue: reduce the complexity. [link](https://leetcode.com/problems/4sum/)

Hint: 4Sum = 1+3Sum 🙃. then use 2 pointer approach in 3Sum.

**Implementation**
```python
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        res = []
        nums.sort()
        for i in range(len(nums)):
            if i == 0 or nums[i] > nums[i-1]:
                diff = target - nums[i]
                threeSums = self.threeSum(nums[i+1:], diff)
                for threeSum in threeSums:
                    res.append([nums[i]] + threeSum)
        return res
                
        
    def threeSum(self, nums, target):
        res = []
        if len(nums) < 3: return res

        for i in range(len(nums) - 2):
            if i > 0 and nums[i] == nums[i-1]: continue
            l, r = i + 1, len(nums) - 1
            while l < r :
                s = nums[i] + nums[l] + nums[r]
                if s == target:
                    res.append([nums[i] ,nums[l] ,nums[r]])
                    l += 1; r -= 1
                    while l < r and nums[l] == nums[l - 1]: l += 1
                    while l < r and nums[r] == nums[r + 1]: r -= 1
                elif s < target :
                    l += 1
                else:
                    r -= 1
        return res 
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

     def reverseList(head: ListNode) -> ListNode:
         prev = None
         while head:
             next_node = head.next
             head.next = prev
             prev = head
             head = next_node

         return prev
    
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

-----------
TEST1: 

Eightfold AI:

* que1: DFS with special jumps
* que2: two pointer with nearest difference between 2 numbers
* que3: kill 2/3 in a stack to find the best and worst time
