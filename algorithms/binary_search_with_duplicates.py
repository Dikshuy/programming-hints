from bisect import bisect_left

n = int(input())
ls = list(map(int, input().split()))

m = int(input())
queries = list(map(int, input().split()))

result = []

def binary_search(ls, q):
    idx = bisect_left(ls, q)
    if idx != len(ls) and ls[idx] == q:
        return idx
    else:
        return -1

for q in queries:
    result.append(binary_search(ls, q))

print(*result)
