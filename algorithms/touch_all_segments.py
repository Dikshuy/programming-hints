n = int(input())

points = []

for _ in range(n):
    points.append(list(map(int, input().split())))

points.sort(key = lambda x: x[1])

i = 0

overlap = []

while i < n:
    curr = points[i]
    while i < n-1 and curr[1] >= points[i+1][0]:
        i += 1
    overlap.append(curr[1])
    i += 1

print(len(overlap))
print(*overlap)