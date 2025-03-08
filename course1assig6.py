while True:
    try:
        num = int(input("Enter a number: "))
        result = 100 / num
        print(f"100 divided by {num} is {result}")
        break
    except ZeroDivisionError:
        print("Oops! You cannot divide by zero.")
    except ValueError:
        print("Invalid input! Please enter a valid number.")


my_list = [1, 2, 3]
my_dict = {"a": 1, "b": 2}

try:
    print(my_list[5])
except IndexError:
    print("IndexError occurred! List index out of range.")

try:
    print(my_dict["c"])
except KeyError:
    print("KeyError occurred! Key not found in the dictionary.")

try:
    result = "hello" + 5
except TypeError:
    print("TypeError occurred! Unsupported operand types.")


try:
    num1 = int(input("Enter the first number: "))
    num2 = int(input("Enter the second number: "))
    result = num1 / num2
except ValueError:
    print("Invalid input! Please enter valid numbers.")
except ZeroDivisionError:
    print("Oops! Cannot divide by zero.")
else:
    print(f"The result is {result}.")
finally:
    print("This block always executes.")
