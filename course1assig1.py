name = "Alex"
age = 25
height = 5.9
print(f"Hey there, my name is {name}! I’m {age} years old and {height} feet tall.")


num1 = 10
num2 = 3
print(f"The sum of {num1} and {num2} is", num1 + num2)
print(f"The difference of {num1} and {num2} is", num1 - num2)
print(f"The product of {num1} and {num2} is", num1 * num2)
print(f"The division of {num1} by {num2} is", num1 / num2)


number = float(input("Enter a number: "))
if number > 0:
    print("This number is positive. Awesome!")
elif number < 0:
    print("This number is negative. Better luck next time!")
else:
    print("Zero it is. A perfect balance!")
