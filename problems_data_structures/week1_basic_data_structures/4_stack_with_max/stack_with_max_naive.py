#python3
import sys

class StackWithMax():
    def __init__(self):
        self.__stack = []

    def Push(self, a):
        if len(self.__stack) == 0:
            self.__stack.append(a)
            self.max = a
        elif a <= self.max:
            self.__stack.append(a)
        else:
            self.__stack.append(2*a - self.max)
            self.max = a
        
    def Pop(self):
        assert(len(self.__stack))
        ele = self.__stack.pop()
        if ele > self.max:
            self.max = 2 * self.max - ele

    def Max(self):
        assert(len(self.__stack))
        return self.max


if __name__ == '__main__':
    stack = StackWithMax()

    num_queries = int(sys.stdin.readline())
    for _ in range(num_queries):
        query = sys.stdin.readline().split()

        if query[0] == "push":
            stack.Push(int(query[1]))
        elif query[0] == "pop":
            stack.Pop()
        elif query[0] == "max":
            print(stack.Max())
        else:
            assert(0)
