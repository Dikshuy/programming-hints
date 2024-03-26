n = int(input())

count = 0

if n == 1:
    print(0)
    print(1)
    exit()

if n == 2:
    print(1)
    print(*[1, 2])
    exit()

if n == 3:
    print(1)
    print(*[1, 3])
    exit()

A = [0 for _ in range(n+1)]

A[1] = 0
A[2] = 1
A[3] = 1

for i in range(4, n+1):
    A[i] = 1 + min(A[i-1], A[i//2]+i%2, A[i//3]+i%3)

print(A[n])

nums = [n]

while n != 1:
    if n%3 == 0 and A[n]-1 == A[n//3]:
        nums.append(n//3)
        n = n//3
    elif n%2 == 0 and A[n]-1 == A[n//2]:
        nums.append(n//2)
        n = n//2
    else:
        nums.append(n-1)
        n = n-1

print(*nums[::-1])