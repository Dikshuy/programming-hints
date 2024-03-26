a = input()
b = input()

dp = [[0 for _ in range(len(b)+1)] for _ in range(len(a)+1)]

for i in range(len(a)+1):
    dp[i][0] = i

for j in range(len(b)+1):
    dp[0][j] = j

for i in range(1, len(a)+1):
    for j in range(1, len(b)+1):
        diff = 0 if a[i-1] == b[j-1] else 1
        dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+diff)
        
print(dp[len(a)][len(b)])