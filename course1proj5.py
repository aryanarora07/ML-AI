import turtle

def factorial(n):
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

def draw_branch(t, length, level):
    if level == 0:
        return
    t.forward(length)
    t.left(30)
    draw_branch(t, length * 0.7, level - 1)
    t.right(60)
    draw_branch(t, length * 0.7, level - 1)
    t.left(30)
    t.backward(length)

while True:
    print("Welcome to the Recursive Artistry Program!")
    print("Choose an option:")
    print("1. Calculate Factorial")
    print("2. Find Fibonacci")
    print("3. Draw a Recursive Fractal")
    print("4. Exit")
    choice = input("> ")

    if choice == "1":
        try:
            num = int(input("Enter a number to find its factorial: "))
            if num < 0:
                print("Please enter a positive integer!")
            else:
                print(f"The factorial of {num} is {factorial(num)}.")
        except ValueError:
            print("Invalid input! Please enter a number.")

    elif choice == "2":
        try:
            num = int(input("Enter the position of the Fibonacci number: "))
            if num < 0:
                print("Please enter a positive integer!")
            else:
                print(f"The {num}th Fibonacci number is {fibonacci(num)}.")
        except ValueError:
            print("Invalid input! Please enter a number.")

    elif choice == "3":
        t = turtle.Turtle()
        t.speed(0)
        t.left(90)
        draw_branch(t, 100, 4)
        turtle.done()

    elif choice == "4":
        print("Goodbye!")
        break

    else:
        print("Invalid choice! Please select 1, 2, 3, or 4.")
