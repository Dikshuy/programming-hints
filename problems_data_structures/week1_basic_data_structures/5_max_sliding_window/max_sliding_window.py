# python3
from collections import deque

def max_sliding_window(sequence, m):
    q = deque()
    for i in range(m):
        while q and sequence[i] >= sequence[q[-1]]:
            q.pop()
        q.append(i)
    maximums = [sequence[q[0]]]

    for i in range(m, len(sequence)):
        while q and q[0] <= i-m:
            q.popleft()
        
        while q and sequence[i] >= sequence[q[-1]]:
            q.pop()
        q.append(i)

        maximums.append(sequence[q[0]])

    return maximums

if __name__ == '__main__':
    n = int(input())
    input_sequence = [int(i) for i in input().split()]
    assert len(input_sequence) == n
    window_size = int(input())

    print(*max_sliding_window(input_sequence, window_size))

