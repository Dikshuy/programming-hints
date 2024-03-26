W, n = list(map(int, input().split()))

weights = list(map(int, input().split()))

value = [[0 for _ in range(n+1)] for _ in range(W+1)]

for w in range(1, W+1):
    for i in range(1, n+1):
        value[w][i] = value[w][i-1]
        if weights[i-1] <= w:
            val = value[w-weights[i-1]][i-1] + weights[i-1]
            if val > value[w][i]:
                value[w][i] = val

print(value[W][n])