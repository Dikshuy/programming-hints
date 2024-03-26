money = int(input())

dp = [0 for _ in range(money+1)]

if money == 1 or money == 3 or money == 4:
    print(1)
    exit()

if money == 2:
    print(2)
    exit()

dp[1] = dp[3] = dp[4] = 1
dp[2] = 2

for i in range(5, len(dp)):
    dp[i] = 1 + min(dp[i-1], dp[i-3], dp[i-4])

print(dp[money])