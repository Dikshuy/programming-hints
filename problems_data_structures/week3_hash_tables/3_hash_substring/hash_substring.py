# python3

def print_occurrences(output):
    print(' '.join(map(str, output)))

def poly_hash(s, x, p):
    ans = 0
    for c in reversed(s):
        ans = (ans * x + ord(c)) % p
    return ans

def pre_computed_hash(T, P, x, p):
    lT = len(T)
    lP = len(P)
    H = [[] for _ in range(lT-lP+1)]
    S = T[lT-lP:]
    H[lT-lP] = poly_hash(S, x, p)
    y = 1
    for _ in range(1, lP+1):
        y = (y*x)%p
    for i in range(lT-lP-1, -1, -1):
        H[i] = (x*H[i+1] + ord(T[i]) - y*ord(T[i+lP]))%p
    
    return H

def RabinKarp(T, P):
    p = 1000000007
    x = 263
    positions = []
    pHash = poly_hash(P, x, p)
    H = pre_computed_hash(T, P, x, p)
    for i in range(len(T)-len(P)+1):
        if pHash == H[i]:
            positions.append(i)
    return positions

if __name__ == '__main__':
    P = input()
    T = input()
    print_occurrences(RabinKarp(T, P))