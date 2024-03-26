n = int(input())

ls = list(map(int, input().split()))

major_ele = n // 2

dic = {}

for i in ls:
    if i in dic.keys():
        dic[i] += 1
    else:
        dic[i] = 1

for key, val in dic.items():
    if val > major_ele:
        print(1)
        exit()

print(0)