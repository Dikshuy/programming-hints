n = int(input())

nums = list(input().split())

for i in range(n-1):
    for j in range(i+1, n):
        if nums[i]+nums[j] < nums[j]+nums[i]:
            nums[i], nums[j] = nums[j], nums[i]
        
print(str(int("".join(nums))))