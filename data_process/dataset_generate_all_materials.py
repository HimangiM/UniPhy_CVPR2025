import os

print ('Generating elastic dataset')
os.system('python elastic.py')

print ('Generating newtonian dataset')
os.system('python newtonian.py')

print ('Generating plasticine dataset')
os.system('python plasticine.py')

print ('Generating sand dataset')
os.system('python sand.py')

print ('Generating non_newtonian dataset')
os.system('python non_newtonian.py')
