## Some common problem related to trees

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

### Validate Binary search tree
Problem [link](https://leetcode.com/problems/validate-binary-search-tree/)

Hint: Use recursion call stack

**Implementation**
```python
class Solution(object):
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        return self.validate(root, float('-inf'), float('inf'))

    def validate(self, tree, minimum, maximum):
        if tree == None: return True

        # validate
        if tree.val >= maximum or tree.val <= minimum: return False

        return self.validate(tree.left, minimum=minimum, maximum=tree.val) and self.validate(tree.right, minimum=tree.val, maximum=maximum)
```