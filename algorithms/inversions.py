n = int(input())

ls = list(map(int, input().split()))

def count_inversions(ls):
    global total_inversions

    if len(ls) <= 1:
        return ls

    mid = len(ls) // 2

    left_arr = count_inversions(ls[:mid])
    right_arr = count_inversions(ls[mid:])

    sorted_arr, count = merge(left_arr, right_arr)

    total_inversions += count

    return sorted_arr

def merge(left_arr, right_arr):
    i, j, count = 0, 0, 0
    sorted_arr = []
    
    while i < len(left_arr) and j < len(right_arr):
        if left_arr[i] <= right_arr[j]:
            sorted_arr.append(left_arr[i])
            i += 1
        else:
            sorted_arr.append(right_arr[j])
            count += len(left_arr) - i
            j += 1
        
    sorted_arr += left_arr[i:]
    sorted_arr += right_arr[j:]
    
    return sorted_arr, count

total_inversions = 0
count_inversions(ls)

print(total_inversions)