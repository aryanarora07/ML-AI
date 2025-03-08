text = "Python is amazing!"
print("First word:", text[0:6])
print("Amazing part:", text[10:17])
print("Reversed string:", text[::-1])



text = " hello, python world! "
print(text.strip())
print(text.capitalize())
print(text.replace("world", "universe"))
print(text.upper())


word = input("Enter a word: ")
if word == word[::-1]:
    print(f"Yes, '{word}' is a palindrome!")
else:
    print(f"No, '{word}' is not a palindrome!")
