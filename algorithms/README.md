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
