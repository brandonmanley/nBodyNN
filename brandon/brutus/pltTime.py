import matplotlib.pyplot as plt

times = open("times.txt", "r")

time_values_con = []
time_values_div = []

for line in times:
    print(line.split())
