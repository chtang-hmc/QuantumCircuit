import numpy as np
import matplotlib.pyplot as plt
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
    
    def neighbor(self, node):
        return node in self.neighbors()
    
    def degree(self):
        return len(self.edges)
    
    def add_edge(self, v, weight):
        self.edges.append(Edge(self, v, weight))
        
    def delete_edge(self, v):
        for edge in self.edges:
            if edge.v == v:
                self.edges.remove(edge)
                break
    
    def adj(self, v):
        for edge in self.edges:
            if edge.v == v:
                return edge.weight
        return float('inf')
    
    def set_weight(self, v, weight):
        for edge in self.edges:
            if edge.v == v:
                edge.set_weight(weight)
                break
    
    def coef(self, v):
        return np.exp(-1j*self.adj(v))
    
    def prob(self, v):
        return np.abs(self.coef(v))**2
    
    def __str__(self):
        return f'Node: {self.edges} edges'
    
    def __repr__(self):
        return f'Node: {self.edges} edges'

class Qubit(Node):
    def __init__(self, state = 0):
        self.state = state
        self.edges = []
        
    def __str__(self):
        return f'Qubit: {self.edges} edges'
        
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
    
    def set_weight(self, weight):
        self.weight = weight
        
    def far(self):
        return self.weight == float('inf')
        
class Graph:
    def __init__(self, size):
        self.root = Root()
        self.size = size
        self.nodes = [self.root] + [Qubit() for _ in range(size)] + [Qubit(1) for _ in range(size)]
        self.edges = []
        
    def __str__(self) -> str:
        return f'Graph: {self.nodes} nodes and {self.edges} edges'
    
    def __getitem__(self, i):
        return self.nodes[2*i -1], self.nodes[2*i]
    
    def _getindex(self, node):
        return self.nodes.index(node)
                
    def other(self, node):
        if isinstance(node, Node):
            node = self._getindex(node)
        assert node <= self.size * 2
        if node <= self.size:
            return self.nodes[node + self.size]
        else:
            return self.nodes[node - self.size]
        
    def add_edge(self, u, v, weight):
        new_edge = Edge(u, v, weight)
        u.edges.append(new_edge)
        self.edges.append(new_edge)
        
    def delete_edge(self, u = None, v = None, edge = None):
        if edge:
            edge.u.edges.remove(edge)
            self.edges.remove(edge)
        else:
            for edge in u.edges:
                if edge.v == v:
                    u.edges.remove(edge)
                    self.edges.remove(edge)
                    break
        
    