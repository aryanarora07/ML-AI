password = input("Enter a password: ")
has_upper = False
has_lower = False
has_digit = False
has_special = False
special_chars = "@#$%^&*!"
score = 0

if len(password) >= 8:
    score += 2
else:
    print("Your password needs to be at least 8 characters.")

for char in password:
    if char.isupper():
        has_upper = True
    elif char.islower():
        has_lower = True
    elif char.isdigit():
        has_digit = True
    elif char in special_chars:
        has_special = True

if has_upper:
    score += 2
else:
    print("Your password needs at least one uppercase letter.")
if has_lower:
    score += 2
else:
    print("Your password needs at least one lowercase letter.")
if has_digit:
    score += 2
else:
    print("Your password needs at least one digit.")
if has_special:
    score += 2
else:
    print("Your password needs at least one special character.")

if score == 10:
    print("Your password is strong! 💪")
print(f"Password strength score: {score}/10")
