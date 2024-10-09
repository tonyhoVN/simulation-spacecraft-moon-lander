import matplotlib.pyplot as plt

# Step 1: Create the initial figure with one point
plt.ion()  # Turn on interactive mode
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')

initial_point = ax.plot(1, 1, 'ro')  # Red point at (1, 1)
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.draw()  # Draw the figure

# Step 2: Add a new point dynamically
new_point = ax.plot(3, 4, 'bo')  # Blue point at (3, 4)
plt.draw()  # Update the figure

plt.pause(1)  # Pause to display the plot, so you can see it

# Step 3: Continue adding more points if needed
another_point = ax.plot(7, 8, 'go')  # Green point at (7, 8)
plt.draw()  # Update the figure
plt.pause(1)  # Pause to display