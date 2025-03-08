start = int(input("Enter the starting number: "))
while start > 0:
    print(start, end=" ")
    start -= 1
print("Blast off!")



num = int(input("Enter a number: "))
for i in range(1, 11):
    print(f"{num} x {i} = {num * i}")


num = int(input("Enter a number: "))
factorial = 1
for i in range(1, num + 1):
    factorial *= i
print(f"The factorial of {num} is {factorial}.")
