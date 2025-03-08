import random
number_to_guess = random.randint(1, 100)
attempts = 0
max_attempts = 10

while attempts < max_attempts:
    guess = int(input("Guess the number (between 1 and 100): "))
    attempts += 1
    if guess > number_to_guess:
        print("Too high! Try again.")
    elif guess < number_to_guess:
        print("Too low! Try again.")
    else:
        print(f"Congratulations! You guessed it in {attempts} attempts!")
        break
if attempts == max_attempts:
    print("Game over! Better luck next time!")
