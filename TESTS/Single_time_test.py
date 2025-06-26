import math

def calculate_root_square(a, c):
    if c <= a:
        raise ValueError("c must be greater than a")
    return math.sqrt(c * c - a * a)

x = 348 + 53
y = 50
deg = math.degrees(math.atan(y/x))
print(deg)

x1 = 53 * math.cos(math.radians(deg))
y1 = 53 * math.sin(math.radians(deg))
print("x1:", x1)
print("y1:", y1)
x2 = x - x1
y2 = y - y1

print("x2:", x2)
print("y2:", y2)

