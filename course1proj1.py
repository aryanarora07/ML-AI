age = int(input("How old are you? "))
if age >= 18:
    print("Congratulations! You are eligible to vote. Go make a difference!")
else:
    years_left = 18 - age
    print(f"Oops! You’re not eligible yet. But hey, only {years_left} more years to go!")
