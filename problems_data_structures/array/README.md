## Problems related to arrays

### Subarray with a given sum

Issue: reduce time complexity to O(N). [problem link](https://practice.geeksforgeeks.org/problems/subarray-with-given-sum-1587115621/1#)

Hint: Use sliding window approach

**Implementation**
```python
class Solution:
    def subArraySum(self,arr, n, s): 
       #Write your code here
        current_sum = arr[0]
        start = 0
        end = 1
        while end <= n:
            while current_sum > s and start < end-1:
                current_sum -= arr[start]
                start += 1
            if current_sum == s:
                return [start+1, end]
            if end < n:
                current_sum += arr[end]
            end += 1
        return [-1]
```

### Kadane's algorithm / Max sum of contiguous subarray

Issue: [problem link](https://practice.geeksforgeeks.org/problems/kadanes-algorithm-1587115620/1#)

Hint: use DP

**Implementation**
```python
class Solution:
    def maxSubArraySum(self,arr,N):
        ##Your code here
        s = 0
        res = []
        for i in arr:
            s = max(s+i, i)
            res.append(s)
        return max(res)
```
