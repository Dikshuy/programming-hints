n = int(input())

if n==0:    print(0)

curr, prev = 0, 1

prev, curr = 0, 1
for i in range(0, n-1):
    prev, curr = curr, (prev+curr)

print(curr)