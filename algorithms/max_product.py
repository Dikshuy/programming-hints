n = int(input())

ls1 = list(map(int, input().split()))
ls2 = list(map(int, input().split()))

ls1.sort()
ls2.sort()

ans = 0

for i in range(n):
    ans += ls1[i]*ls2[i]

print(ans)