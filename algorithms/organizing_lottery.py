n, m = list(map(int, input().split()))
starts, ends = [], []
for _ in range(n):
    start, end = list(map(int, input().split()))
    starts.append(start)
    ends.append(end)

points = list(map(int, input().split()))


def points_cover_fast(starts, ends, points):
    count = [0] * len(points)

    li = [(i, 's') for i in starts] + [(i, 'e') for i in ends] + [(p, 'p', i) for i, p in enumerate(points)]
    li.sort(key=lambda x: (x[0], -ord(x[1])))

    cnt = 0
    for i, tup in enumerate(li):
        if tup[1] == 's':
            cnt += 1
        if tup[1] == 'p':
            count[tup[2]] = cnt;
        if tup[1] == 'e':
            cnt -= 1

    return count

result = points_cover_fast(starts, ends, points)

print(*result)