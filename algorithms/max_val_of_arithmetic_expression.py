inp = input()

inp_len = len(inp)

digits = []
operations = []
i = 0

while i < inp_len-1:
    digits.append(int(inp[i]))
    operations.append(inp[i+1])
    i += 2

digits.append(int(inp[-1]))

n = len(digits)

m = [[0 for _ in range(n)] for _ in range(n)]
M = [[0 for _ in range(n)] for _ in range(n)]

def calculate(a, b, opr):
    if opr == "+":
        return a + b
    elif opr == "-":
        return a - b
    else:
        return a * b

def min_and_max(i, j):
    min_ = 10**8
    max_ = -10**8

    for k in range(i, j):
        a = calculate(M[i][k], M[k+1][j], operations[k])
        b = calculate(M[i][k], m[k+1][j], operations[k])
        c = calculate(m[i][k], M[k+1][j], operations[k])
        d = calculate(m[i][k], m[k+1][j], operations[k])

        min_ = min(min_, a, b, c, d)
        max_ = max(max_, a, b, c, d)

    return min_, max_

for i in range(n):
    m[i][i] = digits[i]
    M[i][i] = digits[i]

for s in range(1, n):
    for i in range(0, n-s):
        j = i + s
        m[i][j], M[i][j] = min_and_max(i, j)

print(M[0][n-1])