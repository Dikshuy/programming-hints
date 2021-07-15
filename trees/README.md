## Some common problem related to trees

---
### Maximum depth of binary trees

Problem [link](https://leetcode.com/explore/featured/card/top-interview-questions-easy/94/trees/555/)

Hint: Use recursion

**Implementation**
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if root == None: return 0
        
        if root.left == None and root.right == None:
            return 1
        
        return max(self.maxDepth(root.left), self.maxDepth(root.right))+1
```