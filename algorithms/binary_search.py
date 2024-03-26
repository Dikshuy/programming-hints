n = int(input())
ls = list(map(int, input().split()))

m = int(input())
queries = list(map(int, input().split()))

def binary_search(ls, q):
    left = 0
    right = n-1
    while right >= left:
        mid = (left + right) // 2
        if q == ls[mid]:
            return mid
        elif q > ls[mid]:
            left = mid + 1
        else:
            right = mid - 1
    return -1

result = []

for q in queries:
    result.append(binary_search(ls, q))

print(*result)
