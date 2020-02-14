"""
- Generate a random file name
    - File name should use datetime
    - Have param for specific name
- open file of that name
- Create random number of rows
- Function will have certain datatypes it can do 
    - Name
    - Age
    - Education type
    - random floats
    - random ints
- each row should have a pid
"""

from faker import Faker
import time
import datetime
import random
import re

def generate(name=None,rows=50, headers=["Name","Age", "Education", "GPA"], datatypes=["name", "i100", "edu", "f5"]):
    if len(headers) != len(datatypes):
        print("Need a datatype for each header")
        return
    fake = Faker()
    edu_levels = ["Primary", "Secondary", "Bachelors", "Masters", "PhD"]
    if not name:
        filename = str(int(time.time()))
    else:
        filename = name
    filename += ".csv"
    f = open(filename, "w")
    headers.insert(0,"pid")
    pid = 1
    f.write(','.join(headers)+"\n")

    for pid in range(0, rows):
        row = [str(pid)]
        for i in range(0, len(datatypes)):
            if datatypes[i] == "name":
                row.append(fake.name())
            elif datatypes[i] == "edu":
                row.append(edu_levels[(random.randrange(100)%6)-1])
            elif re.search("([i][0-9]*)",datatypes[i]):
                row.append(str(random.randrange(int(datatypes[i].split("i")[1]))))
            elif re.search("([f][0-9]*)",datatypes[i]):
                row.append(str(random.uniform(0,int(datatypes[i].split("f")[1]))))
        f.write(','. join(row)+"\n")
    return filename

if __name__ == "__main__":
    filename = generate(name="test")
    print(filename)


    

        

    
