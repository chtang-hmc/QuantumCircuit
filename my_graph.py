import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy as sp
import quantum_graph as qg

class Qubits(qg.Graph):
    def __init__(self, size):
        super().__init__(size)
        for i in range(size):
            self.add_edge(self.root, self.nodes[i+1], 0)
            for j in range(i, size):
                self.add_edge(self.nodes[i+1], self.nodes[j+1], 0)
                self.add_edge(self.nodes[j+1], self.nodes[i+1], 0)
        for node in self.nodes:
            self.add_edge(node, node, 0)