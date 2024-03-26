def count_sort(ls):
    m = max(ls)

    count_ls = [0] * (m+1)

    for num in ls:
        count_ls[num] += 1

    for i in range(1, m+1):
        count_ls[i] += count_ls[i-1]

    output_ls = [0] * len(ls)

    for i in range(len(ls)-1, -1, -1):
        output_ls[count_ls[ls[i]]-1] = ls[i]
        count_ls[ls[i]] -= 1

    return output_ls

if __name__ == "__main__":
    input_ls = [0,3,2,4,5,1,2,3,7,1,9,10,0,1]

    output = count_sort(input_ls)

    for num in output:
        print(num, end=" ")