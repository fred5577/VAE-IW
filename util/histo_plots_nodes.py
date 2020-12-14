import csv
import matplotlib.pyplot as plt
import statistics

def create_histo_plot(domain):
    with open('number_of_nodes_0.5.csv', 'r') as file:
        reader = csv.reader(file)
        counter = 0
        for row in reader:
            if row[0] == domain:
                results = list(map(int, row[1:]))
                plt.hist(results, bins = 10)
                plt.savefig("histo_" + domain + ".png")

def get_node_average_for_each_domain():
    with open('number_of_nodes_0.5.csv', 'r') as file:
        reader = csv.reader(file)
        skip_first = True
        for row in reader:
            if skip_first:
                skip_first = False
                continue
            results = list(map(int, row[1:]))
            mean = sum(results[1:])/len(results[1:])
            print(row[0] + " mean nodes: " + str(mean) + " std. " + str(statistics.stdev(results[1:]))) 

def get_depth_average_for_each_domain():
    with open('depth_of_tree_0.5.csv', 'r') as file:
        reader = csv.reader(file)
        skip_first = True
        for row in reader:
            if skip_first:
                skip_first = False
                continue
            results = list(map(int, row[1:]))
            mean = sum(results[1:])/len(results[1:])
            print(row[0] + " depth nodes: " + str(mean) + " std. " + str(statistics.stdev(results[1:])))    

create_histo_plot("Asterix")

get_node_average_for_each_domain()

get_depth_average_for_each_domain()