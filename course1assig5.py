def greet_user(name):
    print(f"Hello, {name}! Welcome aboard.")

def add_numbers(num1, num2):
    return num1 + num2

greet_user("Alice")
print(f"The sum of 5 and 10 is {add_numbers(5, 10)}.")


def describe_pet(pet_name, animal_type="dog"):
    print(f"I have a {animal_type} named {pet_name}.")

describe_pet("Buddy")
describe_pet("Whiskers", "cat")


def make_sandwich(*ingredients):
    print("Making a sandwich with the following ingredients:")
    for ingredient in ingredients:
        print(f"- {ingredient}")

make_sandwich("Lettuce", "Tomato", "Cheese")


def factorial(n):
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print(f"Factorial of 5 is {factorial(5)}.")
print(f"The 6th Fibonacci number is {fibonacci(6)}.")
