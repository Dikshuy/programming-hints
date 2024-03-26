"""
Problem:
You are given a list of n integers, and your task is to calculate the number of distinct values in the list.
Input
The first input line has an integer n: the number of values.
The second line has n integers x_1,x_2,.......,x_n.
Output
Print one integers: the number of distinct values.
Constraints

1 <= n <= 2.10^5
1 <= x_i <= 10^9
"""

n = input()
ls = list(map(int, input().strip().split()))

print(len(set(ls)))