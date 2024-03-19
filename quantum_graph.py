import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy as sp

class Qubit:
    def __init__(self, state = 0):
        self.state = state
        self.edges = []
        
    def __getitem__(self, i):
       if self.edges[i]:
           return self.edges[i].weight
       else:
           return float('inf')
        
class Root(Qubit):
    def __init__(self, state = 0):
        super().__init__(state)
        
class Edge:
    def __init__(self, u, v, weight):
        self.u = u
        self.v = v
        self.weight = weight
        
    def __str__(self):
        return f'({self.u}, {self.v}, {self.weight})'
        
class Graph:
    def __init__(self, size):
        self.root = Root()
        self.size = size
        self.nodes = [Qubit() for _ in range(size)] + [Qubit(1) for _ in range(size)]
        self.edges = []
        
    def add_edge(self, u, v, weight):
        new_edge = Edge(u, v, weight)
        u.edges.append(new_edge)
        self.edges.append(new_edge)
        
    