import os 
parent_path = os.path.abspath(os.path.dirname('__file__'))
save_path = os.path.join(parent_path,"data","scraft.csv")
test = os.getcwd()
print(test)

step_size=1
for i in range(0, 10, step_size):
    if i>3: step_size=2
    print(i)