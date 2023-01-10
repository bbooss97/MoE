#read the data folder and create a file with the annotations
import os 
import classes

with open("./annotations.txt", "w") as f:
    for i,n in enumerate(classes):
        for file in os.listdir("./Data/Data/"+n):
            f.write("./Data/Data/"+n+"/"+file+" "+str(i)+"\n")
    print(n+" done")