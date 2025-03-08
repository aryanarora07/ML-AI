inventory = {}

inventory["apple"] = (10, 2.5)
inventory["banana"] = (20, 1.2)

inventory["mango"] = (15, 3.0)
del inventory["apple"]
inventory["banana"] = (25, 1.2)

print("Welcome to the Inventory Manager!")
print("Current inventory:")
for item, (quantity, price) in inventory.items():
    print(f"Item: {item}, Quantity: {quantity}, Price: ${price}")
print("Adding a new item: mango")
print("Updated inventory:")
for item, (quantity, price) in inventory.items():
    print(f"Item: {item}, Quantity: {quantity}, Price: ${price}")

total_value = 0
for quantity, price in inventory.values():
    total_value += quantity * price
print(f"Total value of inventory: ${total_value}")
