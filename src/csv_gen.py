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

def create_file(name=None):
    if not name:
        filename = str(int(time.time()))
    else:
        filename = name
    filename+=".csv"
    f = open(filename, "w")
    return f, filename

def gen_rand_number(dt):
    if re.search("(int)([0-9]*)",dt):
        return random.randrange(int(dt.split("int")[1]))
    elif re.search("(float)([0-9]*)",dt):
        return random.uniform(0,int(dt.split("float")[1]))
    else:
        return "0"


def generate(name=None,rows=50, headers=["Name","Age", "Education", "GPA"], datatypes=["name", "int100", "edu", "float5"]):
    if len(headers) != len(datatypes):
        print("Need a datatype for each header")
        return
    fake = Faker()
    edu_levels = ["Primary", "Secondary", "Bachelors", "Masters", "PhD"]
    f, filename = create_file(name)
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
            elif re.search("(int|float)", datatypes[i]):
                row.append(str(gen_rand_number(datatypes[i])))

        f.write(','. join(row)+"\n")
    return filename

def generate_output_data(name=None, rows=50, headers=["Name", "Age", "Education", "GPA"], datatypes=["name", "int100", "edu", "float5"], generalise=["Age", "GPA"]):
    if len(headers) != len(datatypes):
        print("Need a datatype for each header")
        return
    fake = Faker()
    edu_levels = ["Primary", "Secondary", "Bachelors", "Masters", "PhD"]
    f, filename = create_file(name)
    temp = []
    temp.append("pid")
    pid = 1
    for i in range(0, len(headers)):
        # In this loop, each header should be check if its in generalise
        # If it is, create a min and a max 
        if headers[i] in generalise:
            temp.append("min"+headers[i])
            temp.append("max"+headers[i])
        else:
            temp.append(headers[i])
    f.write(','.join(temp)+"\n")
    
    for pid in range(0, rows):
        row = [str(pid)]
        for i in range(0, len(headers)):
            if datatypes[i] == "name":
                row.append(fake.name())
            elif datatypes[i] == "edu":
                row.append(edu_levels[(random.randrange(100)%6)-1])
            elif re.search("(int|float)", datatypes[i]):
                max = gen_rand_number(datatypes[i])
                if headers[i] in generalise:
                    min = random.uniform(0, max)
                    if type(max) is int:
                        min = int(min)
                        while min == max:
                            if max == 0:
                                max = 1
                            min = int(random.uniform(0, max))
                            
                    else:
                        while min == max:
                            min = random.uniform(0, max)
                    row.append(str(min))
                row.append(str(max))
        f.write(','. join(row)+"\n")        
    return filename