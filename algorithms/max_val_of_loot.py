n, W = list(map(int, input().split()))

if W == 0:  print(0)

w = [0 for _ in range(n)]
c = [0 for _ in range(n)]
price = [0 for _ in range(n)]

for i in range(n):
    c[i], w[i] = list(map(int, input().split()))
    price[i] = c[i] / w[i]

stolen = 0

while W > 0 and len(price) > 0:
    idx = price.index(max(price))
    amount = min(w[idx], W)
    stolen += amount * price[idx]
    W -= amount
    price.remove(price[idx])

print(f"{stolen:0.4f}")