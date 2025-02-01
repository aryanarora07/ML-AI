# a = 10

# b = 20

# print(a+b)


# for i in range(0, 14, 2):            range() is commonly used with for loops for a controlled number of iterations.
#     print(i)





## while loops - used when u dont know the no of times for the loop to repeat

# count = 5

# while count>0:
#     print(count)
#     count -= 1


# nested loops
# for i in range(4):
#     for j in range (2):
#         print(j, end = " ") # adding spaces b/w numbers

for num in range(10, 16):

    if num % 3 == 0:

        continue

    if num == 14:

        break

    print(num, end=" ")