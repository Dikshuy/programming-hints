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

### Shortest path in DAG | Topological sort
Issue: there are n planets and n teams are going to complete in the tournament which are numbered from 1 to n, the tournament is going to be hosted on the planet number n. the planets are interconnected via teleportation gateways. the team from planet i can teleport directly to planets i+distance[i] and i-distance[i], only proviede that planets with those numbers exist. one direct teleportation lasts 1 day and teleportation channels have unlimited capacity which means that at any time many teams can be passing from one planet to another. Figure out how many days before the tournament should hey leave from their home planets so they reach just in time. if there is a team which can't reach planet n, the answer for that team would be -1. last question of [problem link](https://drive.google.com/file/d/1JWhfROT25kgXA85OB_nhPkALmzR0XBJc/view)

Hint: find cost-of-shortest-path-in-dag-using-one-pass-of-bellman-ford. set N-1 as the source and perform top sort from there. [blog](https://www.techiedelight.com/cost-of-shortest-path-in-dag-using-one-pass-of-bellman-ford/)

```python
import sys

class Edge:
    def __init__(self, source, dest, weight):
        self.source = source
        self.dest = dest
        self.weight = weight

class Graph:
    def __init__(self, edges, N):
        self.adjList = [[] for _ in range(N)]
        for edge in edges:
            self.adjList[edge.source].append(edge)
 
def DFS(graph, v, discovered, departure, time):
    discovered[v] = True

    for edge in graph.adjList[v]:
        u = edge.dest
        if not discovered[u]:
            time = DFS(graph, u, discovered, departure, time)
    departure[time] = v
    time = time + 1
 
    return time
 
def findShortestDistance(graph, source, N):
    departure = [-1] * N
    discovered = [False] * N
    time = 0
    for i in range(N):
        if not discovered[i]:
            time = DFS(graph, i, discovered, departure, time)
 
    cost = [sys.maxsize] * N
    cost[source] = 0
    for i in reversed(range(N)):
        v = departure[i]
        for e in graph.adjList[v]:
            u = e.dest
            w = e.weight
            if cost[v] != sys.maxsize and cost[v] + w < cost[u]:
                cost[u] = cost[v] + w
    result = []
    for i in range(N - 1):
        if cost[i] == sys.maxsize:  result.append(-1)
        else:   result.append(cost[i])
    
    return result
 
if __name__ == '__main__':
    N = int(input())
    distance = []
    for _ in range(N):  distance.append(int(input()))
    edges = []
    for i in range(N):
        if i+distance[i] < N:
            edges.append((Edge(i+distance[i], i, 1)))
        if i-distance[i] >= 0:
            edges.append((Edge(i-distance[i], i, 1)))
    
    graph = Graph(edges, N)
    source = N-1
    
    res = findShortestDistance(graph, source, N)
    res.append(0)
    print(res)
```

### Largest Rectangle in Histogram
Issue: [problem link](https://leetcode.com/problems/largest-rectangle-in-histogram/)

Hint: if the largest rectangle contains at least 1 bar in full then, if we find areas of all largest rectangle for each bar included full then we can find the max rectangle area. thus, we just need to find the largest rectangle including each bar one by one and take the max of all the max areas for each bar. refer this [video](https://www.youtube.com/watch?v=vcv3REtIvEo)

**Implementation**
```python
# Brute Force method
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        area = []
        n=len(heights)
        for i in range(n):
            l = 0
            r = 0
            for j in range(i-1, -1, -1):
                if heights[j] < heights[i]:
                    break
                l -= 1
            for k in range(i+1, n):
                if heights[k] < heights[i]:
                    break
                r += 1
            a = (r-l+1)*heights[i]
            area.append(a)
            
        return max(area)
	
# but this solution is not optimal as its time complexity is O(N^2)
# Better Approach using stack

class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        n=len(heights)		
        # next smaller right side
        def sr(heights,n):
            stack=[0]
            ans=[n] *n
            for i in range(1,n):
                while stack and heights[stack[-1]]>=heights[i]:
                    x=stack.pop()
                    ans[x]=i
                stack.append(i)
            return ans

        # next smaller left side
        def sl(heights,n):
            stack=[n-1]
            ans=[-1] *n
            for i in range(n-2,-1,-1):
                while stack and heights[stack[-1]]>heights[i]:
                    x=stack.pop()
                    ans[x]=i
                stack.append(i)
            return ans

        la=sl(heights,n)
        ra=sr(heights,n)
        width=[]

        for i in range(n):
            width.append(ra[i]-la[i]-1)
        ans=0

        for i in range(n):
            ans=max(ans,width[i]*heights[i])
        return ans
```

### Word Search  II
Issue: search a lists of words in a grid. [problem link](https://leetcode.com/problems/word-search-ii/)

Hint: use DFS similar to problem #1

**Implementation**
```python
### DFS Approach

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
	
# we can improve this code by optmizing DFS using **hash maps**, as we will be able te reduce the DFS start point to initiate a search

# TRIE + DFS Approach

class TrieNode:
    def __init__(self):
        self.children = [None] * 26
        self.end = False
    
class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        
        self.maxWords = len(words)
        
        # Create Trie root
        self.root = TrieNode()
        
        # Add words to Trie
        for word in words:
            self.add(word)
            
        self.res = set()
        self.r = len(board)
        if self.r == 0:
            return list(res)
        self.c = len(board[0])
        if self.c == 0:
            return list(res)
        
        self.visited = [[False] * self.c for _ in range(self.r)]
        
        for i in range(self.r):
            for j in range(self.c):
                idx = ord(board[i][j]) - 97
                if self.root.children[idx]:
                    self.visited[i][j] = True
                    self.dfs(board, i, j, board[i][j], self.root.children[idx])
                    self.visited[i][j] = False
                
        return list(self.res)
    
    def dfs(self, board, i, j, path, trieNode):
        if trieNode.end:
            self.res.add(path)
        if len(self.res) == self.maxWords:
            return

        for x,y in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            x_1, y_1 = i + x, j + y 
            if self.isValid(x_1, y_1):
                idx = ord(board[x_1][y_1]) - 97
                if trieNode.children[idx]:
                    self.visited[x_1][y_1] = True
                    self.dfs(board, x_1, y_1, path + board[x_1][y_1], trieNode.children[idx])
                    self.visited[x_1][y_1] = False
                
    def isValid(self, i, j):
        return i >= 0 and j >= 0 and i < self.r and j < self.c and not self.visited[i][j]
    
    def add(self, word):
        tmp = self.root
        for w in word:
            c = ord(w) - 97
            if not tmp.children[c]:
                tmp.children[c] = TrieNode()
            tmp = tmp.children[c]
        tmp.end = True

NOTE: both TRIE+DFS and HM+DFS will have same time complexity
```

### Stone Game -III
Issue: pick max of 3 stones to win the game given optimal step chosen [problem link](https://leetcode.com/problems/stone-game-iii/)

Hint: use DP. refer to this [video](https://www.youtube.com/watch?v=HsY3jFyaePU)

**Implementation**
```python
# recursive solution
class Solution:
    def stoneGameIII(self, stoneValue: List[int]) -> str:
        def rec(stone, i):
            if i >= len(stone): return 0
            ans = -math.inf
            ans = max(ans, stone[i] - rec(stone, i+1))
            if i+1 < len(stone): ans = max(ans, stone[i]+stone[i+1] - rec(stone, i+2))
            if i+2 < len(stone): ans = max(ans, stone[i]+stone[i+1]+stone[i+2] - rec(stone, i+3))
            return ans
        stones = rec(stoneValue, 0)
        if stones > 0: return "Alice"
        if stones == 0: return "Tie"
        return "Bob"
	
# dynamic programming approaches
# 1. top down (memoization) approach 

class Solution:
    def stoneGameIII(self, stoneValue: List[int]) -> str:
        def rec(stone, i):
            if i >= len(stone): return 0
            if dp[i]!= -1: return dp[i]
            ans = -math.inf
            ans = max(ans, stone[i] - rec(stone, i+1))
            if i+1 < len(stone): ans = max(ans, stone[i]+stone[i+1] - rec(stone, i+2))
            if i+2 < len(stone): ans = max(ans, stone[i]+stone[i+1]+stone[i+2] - rec(stone, i+3))
            dp[i] = ans
            return dp[i] 
        dp = [-1]*50000
        stones = rec(stoneValue, 0)
        if stones > 0: return "Alice"
        if stones == 0: return "Tie"
        return "Bob"

# 2. bottom up (tabulation approach) (Fastest)

class Solution:
    def stoneGameIII(self, stoneValue: List[int]) -> str:
        n = len(stoneValue)
        stoneValue += [0, 0, 0]

        dp = [0] * (n + 3)
        for i in range(n)[::-1]:
            x = stoneValue[i]
            y = x + stoneValue[i + 1]
            z = y + stoneValue[i + 2]
            dp[i] = max(x - dp[i + 1], y - dp[i + 2], z - dp[i + 3])

        if dp[0] > 0: return 'Alice'
        if dp[0] < 0: return 'Bob'
        return 'Tie'
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
### Best Time to Buy and Sell Stock with Cooldown
Issue: How to solve with cooldown condition imposed. [problem link](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/)

Hint: One easy approach will be to use recursion but that will increase oue time complexity to O(2^N). A better approach will be to define 3 different states when we have our stocks in hand, when we don't have any stock and when we want to sell that. There will be certain possibilites to arrive at these states from previous day. For example: we can come to no stocks in hand if the previous day, we sell any stock or we don't have any stock the last day as well. we will take the max of these two. Basically build a **state transition diagram**. Now, at the end we will just compare the last element of the no stock and sell arrays to find out the maximum profit we can generate. Refer this [video](https://www.youtube.com/watch?v=4wNXkhAky3s) for better understanding.

NOTE: this problem can be solved by valley-peak approach if there is no cooldown period. we just need to find out the local minima and maxima and find out the difference between those to find out the max period. [problem link](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/)

**Implementation**
```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if len(prices) <= 1: return 0
        n = len(prices)
        noStock, inHand, sold = [0]*n, [0]*n, [0]*n
        
        noStock[0] = 0
        inHand[0] = -prices[0]
        sold[0] = 0
        for i in range(1, n):
            noStock[i] = max(noStock[i-1], sold[i-1])
            inHand[i] = max(inHand[i-1], noStock[i-1]-prices[i])
            sold[i] = inHand[i] + prices[i]
            
        return max(noStock[n-1], sold[n-1])
```

### Best Time to Buy and Sell Stock III
Issue: only 2 transactions allowed. [problem link](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/)

Hint: Use divide and conquer approach. divide the array in two parts and find individual local minima and maxima. refer this [video](https://www.youtube.com/watch?v=37s1_xBiqH0)

**Implementation**
```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        if n==0: return 0
        left = [0]*n
        right = [0]*n
        l_min = prices[0]
        r_max = prices[n-1]
        for i in range(1, n):
            left[i] = max(left[i-1], prices[i]-l_min)
            l_min = min(l_min, prices[i])
        for i in range(n-2, -1, -1):
            right[i] = max(right[i+1], r_max-prices[i])
            r_max = max(r_max, prices[i])
        profit = right[0]
        for i in range(1, n):
            profit = max(profit, left[i-1]+right[i])
        return profit
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

### Cyclic Shift / Maximum binary number
Issue: reduce time complecity of cyclic shift. [problem link](https://www.hackerearth.com/practice/data-structures/advanced-data-structures/suffix-arrays/practice-problems/algorithm/maximum-binary-number-2980dd7b/)

Hints: 

1. rotate only if (i)th element is '1' and (i-1)th element is not '1'. this will always result in max possible number and no of shifts can also be reduced

2. find out the period of the string using KMP algorithm. create a pi table and period = len(string) - pi[n-1]. 

**Implementation**
```python
'''
# Sample code to perform I/O:

name = input()                  # Reading input from STDIN
print('Hi, %s.' % name)         # Writing output to STDOUT

# Warning: Printing unwanted or ill-formatted data to output will cause the test cases to fail
'''
T = int(input())
for _ in range(T):
    N,K = map(int, input().split())

    A = input()
    B = A
    i = 1
    x = N
    while i<N:
        s = ''
        if A[i] == '1' and A[i-1] != '1':
            s = A[i:N] + A[0:i]
        if s > B:
            B = s
            x = i 
        i += 1
    
    # KMP algorithm can be used to find out period of string
    # generate pi table 
    pi = [0 for _ in range(N)]
    for i in range(1, N):
        j = pi[i-1]
        while j > 0 and B[i] != B[j]:
            j = pi[j-1]
        if B[i] == B[j]:
            j += 1
        pi[i] = j

    period = N - pi[N-1]
    ans = (K-1)*period + 1*x
    if x == N:  print((K-1)*period)
    else:   print(ans)

```

### Largest prime number from subsequence
Issue: find out the largest Prime Number possible from a subsequence of a Binary String

Hint: find out all the subsequences and store if their int form is prime

**Implementation**
```python
arr = []

def isPrime(x):
    if x <= 1: return False
    for i in range(2, x+1):
        if i*i > x: break
        if x%i == 0: return False
    return True

# obtaining all substrings

def subsequence(input, output):
    if len(input) == 0:
        if output != '' and isPrime(int(output,2)):
                arr.append(output)
        return
    subsequence(input[1:], output+input[0])
    subsequence(input[1:], output)
 

if __name__ == '__main__':
    s = input()
    out = ""
    subsequence(s, out)
    max_ = 0
    for i in arr:
        max_ = max(max_, int(i,2))
    if max_ <= 1:  print(-1)
    else:   print(max_)
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

### Maximum length of the repeated subarray
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
