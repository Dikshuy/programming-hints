# python3

import sys, threading
sys.setrecursionlimit(10**6) # max depth of recursion
threading.stack_size(2**27)  # new thread will get stack of such size

class TreeOrders:
  def read(self):
    self.n = int(sys.stdin.readline())
    self.key = [0 for i in range(self.n)]
    self.left = [0 for i in range(self.n)]
    self.right = [0 for i in range(self.n)]
    for i in range(self.n):
      [a, b, c] = map(int, sys.stdin.readline().split())
      self.key[i] = a
      self.left[i] = b
      self.right[i] = c

  def inOrderTraversal(self, result, i):
    if i == -1:  return
    self.inOrderTraversal(result, self.left[i])
    result.append(self.key[i])
    self.inOrderTraversal(result, self.right[i])

  def inOrder(self):
    result = []
    # Finish the implementation
    # You may need to add a new recursive method to do that
    self.inOrderTraversal(result, 0)   
    return result

  def preOrderTraversal(self, result, i):
    if i == -1: return
    result.append(self.key[i])
    self.preOrderTraversal(result, self.left[i])
    self.preOrderTraversal(result, self.right[i])

  def preOrder(self):
    result = []
    # Finish the implementation
    # You may need to add a new recursive method to do that
    self.preOrderTraversal(result, 0)         
    return result

  def postOrderTraversal(self, result, i):
    if i == -1: return
    self.postOrderTraversal(result, self.left[i])
    self.postOrderTraversal(result, self.right[i])
    result.append(self.key[i])

  def postOrder(self):
    result = []
    # Finish the implementation
    # You may need to add a new recursive method to do that
    self.postOrderTraversal(result, 0)     
    return result

def main():
	tree = TreeOrders()
	tree.read()
	print(" ".join(str(x) for x in tree.inOrder()))
	print(" ".join(str(x) for x in tree.preOrder()))
	print(" ".join(str(x) for x in tree.postOrder()))

threading.Thread(target=main).start()
