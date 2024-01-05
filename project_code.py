import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import sys
from collections import deque
from typing import List


# Import dataset
data = pd.read_csv('players_21.csv')
print(data.head())

# Create df
df = data[:1000]
club = df['club_name']
nation = df['nationality']

# Merge clubs and nationalities
merged_list = []
for i in range(len(nation)):
    merged_list.append({club[i], nation[i]})

# Create adjacency list
adjacency_list = [[] for i in range (1000)]
for i in range(1000):
    for j in range(1000):
        if merged_list[i].intersection(merged_list[j]):
            adjacency_list[i].append(j)

# Create a dictionary to map names to ids
name_to_id = {}
counter = 0
for i in df['short_name']:
    name_to_id[i] = counter
    counter += 1

# Create a dictionary to map ids to names
id_to_name = {}
counter = 0
for i in df['short_name']:
    id_to_name[counter] = i
    counter += 1

def min_seperation(edge, u, v, n) -> int:
    visited = [False] * n
    # Initialize a list to store the distances of the players from the starting player u
    distances = [sys.maxsize] * n
    # Initialize a queue to store the players that need to be visited
    q = deque()
    # Enqueue the starting player and set its distance to 0
    q.append(u)
    visited[u] = True
    distances[u] = 0
    # Perform BFS
    while q:
        # Dequeue the player at the front of the queue
        x = q.popleft()
        # Check if the current player is the ending player
        if x == v:
            return distances[v]
        # Enqueue all the unvisited neighbors of the current player
        for neighbor in edge[x]:
            if not visited[neighbor]:
                q.append(neighbor)
                visited[neighbor] = True
                distances[neighbor] = distances[x] + 1
    # If the ending player was not reached, return sys.maxsize
    return sys.maxsize

print(min_seperation(adjacency_list, name_to_id['Cristiano Ronaldo'], name_to_id['L. Messi'], 1000))

def print_path(path: List[int]) -> None:
    """
    Prints the path stored in the list 'path'.
    The path consists of a list of vertex IDs.
    """
    size = len(path)
    for i in range(size):
        print(id_to_name[path[i]], end=" ")

    print()

def is_not_visited(x: int, path: List[int]) -> int:
    """
    Check if vertex 'x' has been visited in the path.
    Returns 1 if the vertex has not been visited, 0 otherwise.
    """
    size = len(path)
    for i in range(size):
        if path[i] == x:
            return 0

    return 1

def find_paths(g: List[List[int]], src: int, dst: int) -> None:
    """
    Finds all paths from 'src' to 'dst' in the graph 'g' with 'v' vertices.
    The function prints the paths and stops when it has found 3 paths.
    """
    # Initialize a queue to store the paths that need to be visited
    q = deque()
    # Initialize a counter to keep track of the number of paths found
    path_count = 0

    # Initialize the first path with the start vertex
    path = []
    path.append(src)
    # Add the path to the queue
    q.append(path.copy())

    while q:
        # Dequeue the path at the front of the queue
        path = q.popleft()
        # Get the last vertex in the path
        last = path[len(path) - 1]

        # If the last vertex is the destination and we have found less than 3 paths, print the path
        if last == dst and path_count < 3:
            print_path(path)
            path_count += 1
            # If we have found 3 paths, return from the function
            if path_count == 3:
                return

        # Iterate over the neighbors of the last vertex in the path
        for i in range(len(g[last])):
            # If the neighbor has not been visited in the current path, add it to the path
            if is_not_visited(g[last][i], path):
                new_path = path.copy()
                new_path.append(g[last][i])
                q.append(new_path)


print(find_paths(adjacency_list, name_to_id['Cristiano Ronaldo'], name_to_id['L. Messi']))

# Create a matrix to represent the connections between the players
Connection_Graph = np.zeros((df.shape[0], df.shape[0]))
# Iterate over the rows and columns of the matrix
for i in range(Connection_Graph.shape[0]):
    for j in range(Connection_Graph.shape[0]):
        # If the players have a common team, set the value at (i, j) to 1
        if merged_list[i].intersection(merged_list[j]):
            Connection_Graph[i][j] = 1

# Convert the matrix to a numpy array and create a graph from it
npmatrix = np.array(Connection_Graph)
G = nx.from_numpy_matrix(npmatrix)

# Print the number of edges in the graph
print(G.number_of_edges())

# Calculate the degree centrality of each player
dc = nx.algorithms.centrality.degree_centrality(G)
# Initialize variables to store the maximum degree centrality and the index of the player with the maximum value
max_dc = dc[0]
max_index = 0
# Iterate over the degree centralities
for i in dc:
    # If the current degree centrality is greater than the maximum, update the maximum and the index
    if dc[i] > max_dc:
        max_dc = dc[i]
        max_index = i
# Print the name of the player with the highest degree centrality
print(id_to_name[max_index])

# Calculate the closeness centrality of each player
cc = nx.algorithms.centrality.closeness_centrality(G)
# Initialize variables to store the maximum closeness centrality and the index of the player with the maximum value
max_cc = 0
max_index_cc = 0
# Iterate over the closeness centralities
for i in cc:
    # If the current closeness centrality is greater than the maximum, update the maximum and the index
    if cc[i] > max_cc:
        max_cc = cc[i]
        max_index_cc = i
# Print the name of the player with the highest closeness centrality
print(id_to_name[max_index_cc])

# Calculate the betweenness centrality of each player
bc = nx.algorithms.centrality.betweenness_centrality(G)
# Initialize variables to store the maximum betweenness centrality and the index of the player with the maximum value
max_bc = bc[0]
max_index_bc = 0
# Iterate over the betweenness centralities
for i in bc:
    # If the current betweenness centrality is greater than the maximum, update the maximum and the index
    if bc[i] > max_bc:
        max_bc = bc[i]
        max_index_bc = i
# Print the name of the player with the highest betweenness centrality
print(id_to_name[max_index_bc])

# Measure the time taken to calculate the average shortest path length
t1 = time.time()
ds = nx.algorithms.shortest_paths.generic.average_shortest_path_length(G)
print("The degree of separation on average between any two players is " + str(ds))
print("The time taken was " + str(time.time() - t1))

# Set the node size and edge width
node_size = 10
edge_width = 0.1

# Removing self-loops
G.remove_edges_from(nx.selfloop_edges(G))

# Draw the graph using the spring layout
pos = nx.spring_layout(G)
nx.draw(G, pos, node_size=node_size, width=edge_width, with_labels=False)

# Show the plot
plt.show()