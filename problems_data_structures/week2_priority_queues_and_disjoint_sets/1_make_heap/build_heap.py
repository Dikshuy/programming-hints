# python3

def SiftDown(i, size, data, swaps):
    min_idx = i
    l = 2*i + 1
    if l < size and data[l] < data[min_idx]:
        min_idx = l
    r = 2*i + 2
    if r < size and data[r] < data[min_idx]:
        min_idx = r

    if i != min_idx:
        data[i], data[min_idx] = data[min_idx], data[i]
        swaps.append([i, min_idx])
        SiftDown(min_idx, size, data, swaps)

    return swaps

def build_heap(data):
    """Build a heap from ``data`` inplace.

    Returns a sequence of swaps performed by the algorithm.
    """
    # The following naive implementation just sorts the given sequence
    # using selection sort algorithm and saves the resulting sequence
    # of swaps. This turns the given array into a heap, but in the worst
    # case gives a quadratic number of swaps.
    #
    # TODO: replace by a more efficient implementation
    swaps = []
    for i in range(len(data)//2-1, -1, -1):
        swaps = SiftDown(i, len(data), data, swaps)
    return swaps


def main():
    n = int(input())
    data = list(map(int, input().split()))
    assert len(data) == n

    swaps = build_heap(data)

    print(len(swaps))
    for i, j in swaps:
        print(i, j)


if __name__ == "__main__":
    main()
