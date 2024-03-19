import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy as sp

class Node:
    def __init__(self):
        self.in_edges = []
        
    def __getitem__(self, i):
        if self.edges[i]:
            return self.edges[i].weight
        else:
            return float('inf')
        
    def neighbors(self):
        return [edge.v for edge in self.edges]
    
    def degree(self):
        return len(self.edges)
    
    def add_edge(self, v, weight):
        self.edges.append(Edge(self, v, weight))
    
    def __str__(self):
        return f'Node: {self.edges} edges'
    
    def __repr__(self):
        return f'Node: {self.edges} edges'

class Qubit(Node):
    def __init__(self, state = 0):
        self.state = state
        self.edges = []
        
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
        self.nodes = [self.root] + [Qubit() for _ in range(size)] + [Qubit(1) for _ in range(size)]
        self.edges = []
        
    def __str__(self) -> str:
        return f'Graph: {self.nodes} nodes and {self.edges} edges'
    
    def __getitem__(self, i):
        return (self.nodes[2*i -1 ], self.nodes[2*i)
        
    def add_edge(self, u, v, weight):
        new_edge = Edge(u, v, weight)
        u.edges.append(new_edge)
        self.edges.append(new_edge)
        
    def delete_edge(self, u, v):
        for edge in u.edges:
            if edge.v == v:
                u.edges.remove(edge)
                self.edges.remove(edge)
                break
                
    def adj
        
    