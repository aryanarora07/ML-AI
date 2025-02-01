import matplotlib.pyplot as plt
import numpy as np

# basic plotting
# xpoints = np.array([1, 2, 6, 8])
# ypoints = np.array([3, 8, 1, 10])

# plt.plot(xpoints, ypoints)
# plt.show()


# multiple lines
# ypoints = np.array([3, 8, 1, 10])
# xpoints = np.array([1, 2, 6, 8])

# plt.plot(xpoints, ypoints)
# plt.show()



# markers
# ypoints = np.array([10, 20, 30, 40])
# xpoints = np.array([1,2,3,4])

# plt.plot(xpoints, ypoints, marker='o')
# plt.show()

# marker|line|color
# ypoints = np.array([3, 8, 1, 10])
# plt.plot(ypoints, 'o:r')
# plt.show()






# plot labels
# x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
# y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

# plt.plot(x, y)

# plt.xlabel("Average Pulse")
# plt.ylabel("Calorie Burnage")
# plt.title("Sports Watch Data")

# plt.show()




#subplots

# x = np.array([0, 1, 2, 3])
# y = np.array([3, 8, 1, 10])

# plt.subplot(1, 2, 1)
# plt.plot(x,y)

# x = np.array([0, 1, 2, 3])
# y = np.array([10, 20, 30, 40])

# plt.subplot(1, 2, 2)
# plt.plot(x,y)

# plt.show()



#scatter plot

# x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
# y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])


# plt.scatter(x, y)
# plt.show()





#bar chart

# x = np.array(["A", "B", "C", "D"])
# y = np.array([3, 8, 1, 10])

# plt.bar(x,y)
# plt.show()




#histogram

# x = np.random.normal(200, 13, 300) # using normal distribution 

# plt.hist(x)
# plt.show()




