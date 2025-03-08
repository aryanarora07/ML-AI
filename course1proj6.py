def get_number_input(prompt):
    while True:
        try:
            value = float(input(prompt))
            return value
        except ValueError:
            print("Invalid input! Please enter a valid number.")
            logging.error("ValueError occurred: Invalid numeric input.")

def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    """Division function with exception handling for zero division."""
    try:
        return a / b
    except ZeroDivisionError:
        print("Oops! Division by zero is not allowed.")
        logging.error("ZeroDivisionError occurred: division by zero.")
        return None

def calculator():
    """Main calculator function with menu and operations."""
    print("Welcome to the Error-Free Calculator!")
    
    while True:
        try:
            # Display menu
            print("\nChoose an operation:")
            print("1. Addition")
            print("2. Subtraction")
            print("3. Multiplication")
            print("4. Division")
            print("5. Exit")
            
            # Get user choice with exception handling
            choice = input("> ")
            
            # Handle exit option
            if choice == '5':
                print("Goodbye!")
                break
            
            # Validate menu choice
            if choice not in ['1', '2', '3', '4']:
                print("Invalid choice! Please select a number between 1 and 5.")
                continue
            
            # Get input numbers
            num1 = get_number_input("Enter the first number: ")
            num2 = get_number_input("Enter the second number: ")
            
            # Perform selected operation
            result = None
            if choice == '1':
                result = add(num1, num2)
                operation = "addition"
            elif choice == '2':
                result = subtract(num1, num2)
                operation = "subtraction"
            elif choice == '3':
                result = multiply(num1, num2)
                operation = "multiplication"
            elif choice == '4':
                result = divide(num1, num2)
                operation = "division"
            
            # Display result if available
            if result is not None:
                print(f"Result of {operation}: {result}")
                
        except Exception as e:
            # Catch any unexpected exceptions
            error_message = f"An unexpected error occurred: {str(e)}"
            print(error_message)
            logging.error(error_message)
            
        finally:
            # This block always executes, we can use it for cleanup if needed
            pass

if __name__ == "__main__":
    calculator()
