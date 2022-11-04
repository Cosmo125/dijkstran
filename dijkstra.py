#import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt

def rango(ini,fin):
    x=[0 for i in range(501)]
    y=(fin-ini)/500
    x[0]=ini
    for i in range(500):
        x[i+1]=float(x[i]+y)
    return x

def f1(x):
    y=[0 for i in range(len(x))]
    for i in range(len(x)):
        y[i]=-521.536319033495 + 598.0040532702287*x[i] - 272.28558403751583*(x[i]**2) + 61.95434818282811*(x[i]**3) - 7.03745023323986*(x[i]**4) + 0.3192233544175078*(x[i]**5)
    return y
def f2(x):
    y=[0 for i in range(len(x))]
    for i in range(len(x)):
        y[i]=29459.28862120867 - 511.8574186871184*x[i] - 3810.008113104351*(x[i]**2) - 459.88276159579385*(x[i]**3) + 349.58058684199875*(x[i]**4) - 7.699102540302368*(x[i]**5) - 0.1947997497681236*(x[i]**6) - 1.9780504147072588*(x[i]**7) + 0.22781811301890942*(x[i]**8)
    return y
def f3(x):
    y=[0 for i in range(len(x))]
    for i in range(len(x)):
        y[i]=10810.16535043732 - 6672.586317638987*x[i] + 1373.3743826553743*(x[i]**2) - 94.21537571466925*(x[i]**3)
    return y
def f4(x):
    y=[0 for i in range(len(x))]
    for i in range(len(x)):
        y[i]=5.830073521297635 - 3.2562656120026854*x[i] + 1.5186597214737085*(x[i]**2) - 0.26855355215974464*(x[i]**3) + 0.01756929956620013*(x[i]**4)
    return y   
def f5(x):
    y=[0 for i in range(len(x))]
    for i in range(len(x)):
        y[i]=-42083.43266740732 + 43856.636313295065*(x[i]) - 17097.08405163007*(x[i]**2) + 2954.547335722472*(x[i]**3) - 190.90925762851327*(x[i]**4)
    return y
def f6(x):
    y=[0 for i in range(len(x))]
    for i in range(len(x)):
        y[i]=4225.753756924889 - 4447.504973881069*x[i] + 1757.4349725293634*(x[i]**2) - 308.65048133170734*(x[i]**3) + 20.32221406593026*(x[i]**4)
    return y
def f7(x):
    y=[0 for i in range(len(x))]
    for i in range(len(x)):
        y[i]=-849.2933169122923 + 814.0118210868114*x[i] - 289.9507861404056*(x[i]**2) + 45.71154036171668*(x[i]**3) - 2.691486657909921*(x[i]**4)
    return y  

def sqrt(x):
  return x**(1/2)

def pow(base,power):
  return base**power

def zoom(x,y): #zoom de datos
    a1 = (x - 9.02)*1000
    a2 = (y - 79.53)*1000
    return [a1,a2]

def euclidean(x1,y1,x2,y2):
  return sqrt(pow((x1-x2),2)+pow((y1-y2),2))

def e_distance(X,Y,x,y):
    #Calculamos las distancias
    e = {i:euclidean(X[i],Y[i],x,y) 
            for i in range(len(X))}
    #Minimizamos
    s2 = dict(sorted(e.items(), key=lambda item: item[1])) #Euclidiana
    #Obtenemos los valores minimos
    val2 = list(s2.keys())[0]
    return val2

import sys
class Graph(object):
    def __init__(self, nodes, init_graph):
        self.nodes = nodes
        self.graph = self.construct_graph(nodes, init_graph)
        
    def construct_graph(self, nodes, init_graph):
        '''
        This method makes sure that the graph is symmetrical. In other words, if there's a path from node A to B with a value V, there needs to be a path from node B to node A with a value V.
        '''
        graph = {}
        for node in nodes:
            graph[node] = {}
        
        graph.update(init_graph)
        
        for node, edges in graph.items():
            for adjacent_node, value in edges.items():
                if graph[adjacent_node].get(node, False) == False:
                    graph[adjacent_node][node] = value
                    
        return graph
    
    def get_nodes(self):
        "Returns the nodes of the graph."
        return self.nodes
    
    def get_outgoing_edges(self, node):
        "Returns the neighbors of a node."
        connections = []
        for out_node in self.nodes:
            if self.graph[node].get(out_node, False) != False:
                connections.append(out_node)
        return connections
    
    def value(self, node1, node2):
        "Returns the value of an edge between two nodes."
        return self.graph[node1][node2]

def dijkstra_algorithm(graph, start_node):
    unvisited_nodes = list(graph.get_nodes())
    # We'll use this dict to save the cost of visiting each node and update it as we move along the graph   
    shortest_path = {}
    # We'll use this dict to save the shortest known path to a node found so far
    previous_nodes = {}
    # We'll use max_value to initialize the "infinity" value of the unvisited nodes   
    max_value = sys.maxsize
    for node in unvisited_nodes:
        shortest_path[node] = max_value
    # However, we initialize the starting node's value with 0   
    shortest_path[start_node] = 0
    # The algorithm executes until we visit all nodes
    while unvisited_nodes:
        # The code block below finds the node with the lowest score
        current_min_node = None
        for node in unvisited_nodes: # Iterate over the nodes
            if current_min_node == None:
                current_min_node = node
            elif shortest_path[node] < shortest_path[current_min_node]:
                current_min_node = node
                
        # The code block below retrieves the current node's neighbors and updates their distances
        neighbors = graph.get_outgoing_edges(current_min_node)
        for neighbor in neighbors:
            tentative_value = shortest_path[current_min_node] + graph.value(current_min_node, neighbor)
            if tentative_value < shortest_path[neighbor]:
                shortest_path[neighbor] = tentative_value
                # We also update the best path to the current node
                previous_nodes[neighbor] = current_min_node
 
        # After visiting its neighbors, we mark the node as "visited"
        unvisited_nodes.remove(current_min_node)
    
    return previous_nodes, shortest_path

def getResult(previous_nodes, shortest_path, start_node, target_node):
    path = []
    node = target_node
    
    while node != start_node:
        path.append(node)
        node = previous_nodes[node]
 
    # Add the start node manually
    path.append(start_node)
    
    return path[::-1]

    #creando los puntos del mapa

x1,x2,x3,x4 = rango(3.88,4.92),rango(4.92,4.95),rango(4.82,4.95),rango(3.64,4.82)
x5,x6,x7 = rango(3.58,3.64),rango(3.58,4.02),rango(3.65,4.57)
y1,y2,y3,y4,y5,y6,y7 = f1(x1),f2(x2),f3(x3),f4(x4),f5(x5),f6(x6),f7(x7)

x = (x1+x2+x3+x4+x5+x6+x7)
y = (y1+y2+y3+y4+y5+y6+y7)

#CONEXIONES ENTRE LOS NODOS PARA DJSKTRA
nodes = [f'{i}' for i in range(len(x))]
graph = {node:{} for node in nodes}
#conexiones primeros 2 modelos
for i in range(1,998):
    current = str(i)
    foward = str(i+1)
    back = str(i-1)
    graph[current][back] = euclidean(x[i],y[i],x[i-1],y[i-1])
    graph[current][foward] = euclidean(x[i],y[i],x[i+1],y[i+1])
graph[nodes[999]][nodes[998]],graph[nodes[998]][nodes[999]] = euclidean(x[999],y[999],x[998],y[998]),euclidean(x[999],y[999],x[998],y[998])
#conexion modelo 3
for i in range(1001,1498):
    current = str(i)
    foward = str(i+1)
    back = str(i-1)
    graph[current][back] = euclidean(x[i],y[i],x[i-1],y[i-1])
    graph[current][foward] = euclidean(x[i],y[i],x[i+1],y[i+1])
graph[nodes[1000]][nodes[1001]],graph[nodes[1001]][nodes[1000]] = euclidean(x[1000],y[1000],x[1001],y[1001]),euclidean(x[1000],y[1000],x[1001],y[1001])
graph[nodes[1498]][nodes[1499]],graph[nodes[1499]][nodes[1498]] = euclidean(x[1498],y[1498],x[1499],y[1499]),euclidean(x[1498],y[1498],x[1499],y[1499])
#conexiones modelo 4
for i in range(1501,1998):
    current = str(i)
    foward = str(i+1)
    back = str(i-1)
    graph[current][back] = euclidean(x[i],y[i],x[i-1],y[i-1])
    graph[current][foward] = euclidean(x[i],y[i],x[i+1],y[i+1])
graph[nodes[1500]][nodes[1501]],graph[nodes[1501]][nodes[1500]] = euclidean(x[1500],y[1500],x[1501],y[1501]),euclidean(x[1500],y[1500],x[1501],y[1501])
graph[nodes[1998]][nodes[1999]],graph[nodes[1999]][nodes[1998]] = euclidean(x[1998],y[1998],x[1999],y[1999]),euclidean(x[1998],y[1998],x[1999],y[1999])
#conexion modelo 5
for i in range(2001,2498):
    current = str(i)
    foward = str(i+1)
    back = str(i-1)
    graph[current][back] = euclidean(x[i],y[i],x[i-1],y[i-1])
    graph[current][foward] = euclidean(x[i],y[i],x[i+1],y[i+1])
graph[nodes[2000]][nodes[2001]],graph[nodes[2001]][nodes[2000]] = euclidean(x[2000],y[2000],x[2001],y[2001]),euclidean(x[2000],y[2000],x[2001],y[2001])
graph[nodes[2498]][nodes[2499]],graph[nodes[2499]][nodes[2498]] = euclidean(x[2498],y[2498],x[2499],y[2499]),euclidean(x[2498],y[2498],x[2499],y[2499])
#conexion modelo 6
for i in range(2501,2998):
    current = str(i)
    foward = str(i+1)
    back = str(i-1)
    graph[current][back] = euclidean(x[i],y[i],x[i-1],y[i-1])
    graph[current][foward] = euclidean(x[i],y[i],x[i+1],y[i+1])
graph[nodes[2500]][nodes[2501]],graph[nodes[2501]][nodes[2500]] = euclidean(x[2500],y[2500],x[2501],y[2501]),euclidean(x[2500],y[2500],x[2501],y[2501])
graph[nodes[2998]][nodes[2999]],graph[nodes[2999]][nodes[2998]] = euclidean(x[2998],y[2998],x[2999],y[2999]),euclidean(x[2998],y[2998],x[2999],y[2999])
#conexion modelo 7
for i in range(3001,3498):
    current = str(i)
    foward = str(i+1)
    back = str(i-1)
    graph[current][back] = euclidean(x[i],y[i],x[i-1],y[i-1])
    graph[current][foward] = euclidean(x[i],y[i],x[i+1],y[i+1])
graph[nodes[3000]][nodes[3001]],graph[nodes[3001]][nodes[3000]] = euclidean(x[3000],y[3000],x[3001],y[3001]),euclidean(x[3000],y[3000],x[3001],y[3001])
graph[nodes[3498]][nodes[3499]],graph[nodes[3499]][nodes[3498]] = euclidean(x[3498],y[3498],x[3499],y[3499]),euclidean(x[3498],y[3498],x[3499],y[3499])

#conexion f1f7
graph[nodes[0]][nodes[3045]],graph[nodes[3045]][nodes[0]] = euclidean(x[0],y[0],x[3045],y[3045]),euclidean(x[0],y[0],x[3045],y[3045])
#conexion f2f3
graph[nodes[999]][nodes[1499]],graph[nodes[1499]][nodes[999]] = euclidean(x[999],y[999],x[1499],y[1499]),euclidean(x[999],y[999],x[1499],y[1499])
#conexion f3f4
graph[nodes[1000]][nodes[1999]],graph[nodes[1999]][nodes[1000]] = euclidean(x[1000],y[1000],x[1999],y[1999]),euclidean(x[1000],y[1000],x[1999],y[1999])
#conexion f4f5
graph[nodes[1500]][nodes[2499]],graph[nodes[2499]][nodes[1500]] = euclidean(x[1500],y[1500],x[2499],y[2499]),euclidean(x[1500],y[1500],x[2499],y[2499])
#conexion f5f6
graph[nodes[2500]][nodes[2000]],graph[nodes[2000]][nodes[2500]] = euclidean(x[2500],y[2500],x[2000],y[2000]),euclidean(x[2500],y[2500],x[2000],y[2000])
#conexion f5f7
graph[nodes[2000]][nodes[3000]],graph[nodes[3000]][nodes[2000]] = euclidean(x[2000],y[2000],x[3000],y[3000]),euclidean(x[2000],y[2000],x[3000],y[3000])
#conexion f6f7
graph[nodes[2525]][nodes[3000]],graph[nodes[3000]][nodes[2525]] = euclidean(x[2525],y[2525],x[3000],y[3000]),euclidean(x[2525],y[2525],x[3000],y[3000])
graph[nodes[2500]][nodes[3000]],graph[nodes[3000]][nodes[2500]] = euclidean(x[2500],y[2500],x[3000],y[3000]),euclidean(x[2500],y[2500],x[3000],y[3000])

#Shortest path
puntos = {
    "A":(9.023971833,79.53360733),
    "B":(9.023688833,79.53411917),
    "C":(9.02394,79.53425367),
    "D":(9.0247885,79.53481033),
    "E":(9.024763667,79.5346835),
    "F":(9.023708,79.53426667),
    "G":(9.023649167, 79.53397483),
    "H":(9.023578 ,79.5341335),
    "I":(9.024002833 ,79.53405917)
}
#P = input("Ingrese el punto de inicio: ")
#Pf = input("Ingrese el punto destino: ")
pin = zoom(puntos["A"][0],puntos["A"][1])
pfi = zoom(puntos["D"][0],puntos["D"][1])
inicio = e_distance(x,y,pin[0],pin[1])
final = e_distance(x,y,pfi[0],pfi[1])

g = Graph(nodes, graph)
previous_nodes, shortest_path = dijkstra_algorithm(graph=g, start_node=str(inicio))
index = getResult(previous_nodes, shortest_path, start_node=str(inicio), target_node=str(final))
index = [int(i) for i in index]
Xr = [x[i] for i in index]
Yr = [y[i] for i in index]

plt.plot(x,y,'o')
plt.plot(Xr,Yr,markersize=5,c='r')
plt.annotate("Inicio",(x[inicio],y[inicio]))
plt.plot(x[e_distance(x,y,pin[0],pin[1])],y[e_distance(x,y,pin[0],pin[1])],'o',markersize=10)
plt.annotate("Destino",(x[final],y[final]))
plt.plot(x[e_distance(x,y,pfi[0],pfi[1])],y[e_distance(x,y,pfi[0],pfi[1])],'o',markersize=10)
plt.rcParams["figure.figsize"] = [10,10]
plt.show(block=False)

print(f'Distancia por recorrer: {shortest_path[str(final)]*117.9265754} - # puntos: {len(Xr)}')
