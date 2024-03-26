n, m = list(map(int, input().split()))

if n==0:    print(0)
"""
important pointers:
1. for a given value of N and M >= 2, the series generated with Fi modulo M (for i in range(N)) is periodic -> Pisano period
2. length of a pisano period ranges from 3 to M^2
3. pisano period always starts with 0 1
"""

def pisanoPeriod(m):
    prev, curr = 0, 1
    for i in range(0, m*m):
        prev, curr = curr, (prev+curr)%m
    
        if prev == 0 and curr == 1:
            return i+1


pisano_period = pisanoPeriod(m)

n = n % pisano_period

if n == 0:  
    print(0)    
    exit()
    
curr, prev = 0, 1

prev, curr = 0, 1
for i in range(0, n-1):
    prev, curr = curr, (prev+curr)%m

print(curr%m)