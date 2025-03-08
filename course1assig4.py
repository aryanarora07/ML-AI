fruits = ['apple', 'banana', 'cherry', 'date', 'elderberry']
fruits.append('fig')
fruits.remove('apple')
print("Original list:", ['apple', 'banana', 'cherry', 'date', 'elderberry'])
print("After adding a fruit:", ['apple', 'banana', 'cherry', 'date', 'elderberry', 'fig'])
print("After removing a fruit:", fruits)
print("Reversed list:", fruits[::-1])

info = {"name": "Alice", "age": 25, "city": "Boston"}
info["favorite color"] = "Blue"
info["city"] = "New York"
print("Keys:", end=" ")
for key in info:
    print(key, end=", ")
print("\nValues:", end=" ")
for value in info.values():
    print(value, end=", ")


favorites = ('Inception', 'Bohemian Rhapsody', '1984')
print("Favorite things:", favorites)
try:
    favorites[0] = 'The Matrix'
except TypeError:
    print("Oops! Tuples cannot be changed.")
print("Length of tuple:", len(favorites))
