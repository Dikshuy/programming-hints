## Some common algorithms that I encountered

### Depth First Search(DFS)

#### pseudocode
```python
n = number of nodes in the graph
g = adjancey list representing graph
visited = [False,....] #size n

def dfs(at):
    if visited[at]: return
    visited[at] = True

    neighbors = graph[at]
    for next in neighbors:
        dfs(next)

start_node = 0
dfs(start_node)
```

### Breadth First Search(BFS)
```python
class Solution:
    def shortestPath(self, grid: List[List[int]], k: int) -> int:
        if not grid:
            return 0
        q = deque()
        rows = len(grid)
        cols = len(grid[0])
        directions=[(1,0),(-1,0),(0,-1),(0,1)]
        q.append((0,0))
        visited = set()
        visited.add((0,0))
        ans = 0
        while q:
            for v in range(len(q)):
                i,j = q.popleft()
                if i == rows-1 and j == cols-1:
                    return
                for d in directions:
                    new_i = i + d[0]
                    new_j = j + d[1]
                    if 0 <= new_i < rows and 0 <= new_j < cols:
                        if grid[new_i][new_j]==0 and (new_i,new_j) not in visited:
                            q.append((new_i,new_j))
                            visited.add((new_i,new_j))
            ans += 1
```

### KMP
```python 
# time complexity = O(N)
# this algorithm is used for pattern matching or finding out the period of the string(prefix function). 
# period means if we perform cyclic rotations then after how many shifts we can get our value back

# pattern matching 
# period check using prefix function approach
N = len(arr)
pi = [0 for _ in range(N)]
for i in range(1, N):
    j = pi[i-1]
    while j > 0 and arr[i] != arr[j]:
        j = pi[j-1]
    if arr[i] == arr[j]:
        j += 1
    pi[i] = j

    period = N - pi[N-1]
```
