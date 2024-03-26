n = int(input())

loot = list(map(int, input().split()))

target = sum(loot)%3

if n < 3 or target != 0:
    print(0)
    exit()

count = 0

W = sum(loot) // 3

value = [[0 for _ in range(n+1)] for _ in range(W+1)]

count = 0

for w in range(1, W+1):
    for i in range(1, n+1):
        value[w][i] = value[w][i-1]
        if loot[i-1] <= w:
            val = value[w-loot[i-1]][i-1] + loot[i-1]
            if val > value[w][i]:
                value[w][i] = val
        if value[w][i] == W:    count += 1
    
if count < 3:   print(0)
else:   print(1)