def sum(x, y, z):
    A = x + y + z
    B = 2*x - z
    C = (A,B)
    return C

C = sum(3, 5, 7)
print(C[0])