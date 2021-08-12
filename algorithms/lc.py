# from collections import Counter

# trap = []
# class Solution:
#     def maxArea(self, height) -> int:
#         m = max(height)
#         indices = [i for i, j in enumerate(height) if j == m]

#         water = m*(indices[-1]-indices[0])
#         if len(indices)==1:
#             height[indices[0]] -= 1
#             water = -1
            
#         if water==0:
#             print(max(trap))
#             return(max(trap))
#         trap.append(water)
        
#         for i in indices:
#             height[i] -= 1

#         self.maxArea(height)
    
# if __name__ == '__main__':
#     # height = [1,8,6,2,5,4,8,3,7]
#     height= [1,2,1]
#     Solution().maxArea(height)
#     # print(Solution().maxArea(height))
#     # print(Solution.maxArea(Solution,height))

# def minSetSize(arr) -> int:
#     count = Counter(arr)
#     new_arr = []
#     check = []
#     # print(count)
#     # print(count.most_common)
#     count = sorted(count.items(), key=lambda x: x[1], reverse=True)
#     for i in count:
#         if len(new_arr) < len(arr)/2:
#             new_arr.extend([i[0]]*i[1])
#             check.append(i)
#     return len(check)

# # # arr = [3,3,3,3,5,5,5,2,2,7]
# # arr = [9,77,63,22,92,9,14,54,8,38,18,19,38,68,58,19]
# # # arr = [7,7,7,7,7,7]
# # # arr = [1,9]
# # # arr = [1000,1000,3,7]
# # # arr = [1,2,3,4,5,6,7,8,9,10]
# # print("length of array:", len(arr))
# # print(minSetSize(arr))

# # matrix = [[1,5,9],[10,11,13],[12,13,15]]
# # matrix = [[-5]]
# # arr = []
# # for i in range(len(matrix)):
# #     for j in range(len(matrix[i])):
# #         arr.append(matrix[i][j])
# # print(arr)

# # matrix = [[1,2],[3,4]]
# # r=2
# # c=2
# # arr = []
# # for i in range(len(matrix)):
# #     for j in range(len(matrix[i])):
# #         arr.append(matrix[i][j])
# # print(arr)
# # print(matrix)    
# # for i in range(1,len(arr)-1):
# #     matrix[i//c][c%i] = arr[i]


# # print(matrix)
# # nums = [0,0,1,1,1,2,2,3,3,4]
# # nums=[1,1,2]
# # nums = [0,0,1,1,1,2,2,3,3,4]
# # none_ele = 0
# # for i in range(len(nums)):
# #     for j in range(i+1, len(nums)):
# #         if nums[i] == nums[j]:
# #             nums[j] = None
# #             none_ele += 1

# # for i in range(len(nums)):
# #     if nums[i]==None:
# #         nums.remove(nums[i])
# #         nums.append(None)

# # print(nums)
# # print(none_ele)
# # print(len(nums)-none_ele)

# # nums = [1,2,3,4,5,6,7]
# # k=3 
# # while k>0:
# #     print("nums: ", nums)
# #     val = nums[-1]
# #     nums.pop(-1)
# #     k -= 1
# #     nums.insert(0,val)

# # print(nums)
# # nums = [4,1,2,1,2]
# # counter = Counter(nums)
# # print(Counter(nums).values())
# # print(Counter(nums).keys())
# # for key, value in dict(counter).items():
# #     if value == 1:
# #         print(key)

# # nums1 = [0]
# # m = 0
# # nums2 = [1]
# # n = 1
# # if n == 0:
# #     print(nums1)
# # if m == 0:
# #     nums1 = nums2
# #     print(nums1)
# # else:
# #     for i in range(len(nums1)-n):
# #         nums1[m+i] = nums2[i]
# #     print(nums1.sort())

# # A = [1,2,3,2,1,4]
# # B = [3,2,1,4,7,4]
# # n = len(A)
# # m = len(B)

# # # Auxiliary dp[][] array
# # dp = [[0 for i in range(n + 1)] for i in range(m + 1)]

# # # Updating the dp[][] table
# # # in Bottom Up approach
# # for i in range(n - 1, -1, -1):
# #     for j in range(m - 1, -1, -1):

# #         # If A[i] is equal to B[i]
# #         # then dp[j][i] = dp[j + 1][i + 1] + 1
# #         if A[i] == B[j]:
# #             dp[j][i] = dp[j + 1][i + 1] + 1
# # maxm = 0
# # for i in dp:
# # 	for j in i:
# # 		# Update the length
# # 		maxm = max(maxm, j)

# # print(maxm)

# # nums = [10,9,2,5,3,7,101,18]
# # def LIS(nums):
# # 	L = [1]*len(nums)
# # 	# L[0] = [nums[0]]
# # 	for i in range(1, len(nums)):
# # 		for j in range(i):
# # 			if nums[j] < nums[i] and L[i] < L[j]+1:
# # 				L[i] = L[j]+1
# # 	maximum = 0
# # 	print(L)
# # 	for i in range(len(nums)):
# # 		maximum = max(maximum, L[i])
# # 	return maximum

# # print(LIS(nums))

# # import bisect
# # envelopes = [[5,4],[6,7],[6,4],[2,3]]
# # envelopes = [[1,3],[3,5],[6,7],[6,8],[8,4],[9,5]]
# # envelopes = [[1,1],[1,1],[1,1]]
# # envelopes = [[30,50],[12,2],[3,4],[12,15]]
# # envelopes.sort(key=lambda x: (x[0], -x[1]))
# # count = len(envelopes)
# # print(envelopes)
# # dp = []
# # val = 0
# # for env in envelopes:
# # 	i = bisect.bisect_left(dp, env[1])
# # 	print("i;",i)
# # 	if i == len(dp):
# # 		dp.append(env[1])
# # 	else:
# # 		print(env[1])
# # 		dp[i] = env[1]
# # print(dp)
# # print(len(dp))

# # def maxIncreaseKeepingSkyline(grid):
# # 	l1=len(grid)
# # 	l2=len(grid[0])
# # 	q=[max(grid[i]) for i in range(l1)]
# # 	e=[max([grid[z][j] for z in range(l1)]) for j in range(l2)]     
# # 	c=sum((min(q[i],e[j])-grid[i][j]) for j in range(l2) for i in range(l1))
# # 	return c

# # grid = [[3,0,8,4],[2,4,5,7],[9,2,6,3],[0,3,1,0]]
# # print("output:")
# # print(maxIncreaseKeepingSkyline(grid))

# # nums1=[3,1,2]
# # nums2=[1,1]
# # # print(nums1)
# # # print(nums2)
# # arr = []
# # if len(nums2)>len(nums1):
# # 	for i in nums1:
# # 		if i in nums2:
# # 			arr.append(i)
# # if len(nums2)<len(nums1):
# # 	for i in nums2:
# # 		if i in nums1:
# # 			arr.append(i)
# # if len(nums1)==len(nums2):
# # 	print(list(set(nums1) and set(nums2)))
# # # print(set(arr))
# # # print(arr[:len(nums2)])
# # print(arr)

