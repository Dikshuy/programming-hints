import math

n = int(input())

k = int((math.sqrt(8*n+1)-1)/2)

parts = []

for i in range(1, k):
    parts.append(i)

parts.append(n-(k*(k-1)//2))

print(k)
print(*parts)