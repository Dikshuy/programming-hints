def count_sort(ls, exp):
    m = max(ls)
    n = len(ls)

    output = [0] * n
    count = [0]*10

    for i in range(0, n):
        idx = ls[i] // exp
        count[idx%10] += 1

    for i in range(1, 10):
        count[i] += count[i-1]

    i = n-1
    while i >= 0:
        idx = ls[i] // exp
        output[count[idx%10]-1] = ls[i]
        count[idx%10] -= 1
        i -= 1

    return output

def radix_sort(ls):
    max_num = max(ls)

    exp = 1
    while max_num / exp >= 1:
        ls = count_sort(ls, exp)
        exp *= 10

    return ls


if __name__ == "__main__":
    input_ls = [0,3,2,4,5,1,2,3,7,1,9,10,0,1,2,9,5]

    output = radix_sort(input_ls)

    for num in output:
        print(num, end=" ")