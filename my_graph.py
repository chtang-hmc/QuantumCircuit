import numpy as np
import matplotlib.pyplot as plt
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
            
    def measure_without_change(self, qubit):
        if type(qubit) == int:
            qubit = self.nodes[qubit]
        prob = self.root.prob(qubit)
        return np.random.choice([qubit, qubit.other], p=[prob, 1-prob])
    
    def measure(self, qubit):
        if type(qubit) == int:
            qubit = self.nodes[qubit]
        result = self.measure_without_change(qubit)
        for edge in self.edges:
            if edge.v == qubit:
                edge.set_weight(0)
            if edge.v == qubit.other:
                self.delete_edge(edge)
        return result
    
    def measure_and_change_root(self, qubit):
        if type(qubit) == int:
            qubit = self.nodes[qubit]
        result = self.measure(qubit)
        self.root = result
        return result
    
    def multimeasure(self, qubits):
        results = []
        for qubit in qubits:
            result = self.measure_and_change_root(qubit)
            results.append(result)
        return results
    
    def weights(self, qubit):
        if type(qubit) == int:
            qubit = self.nodes[qubit]
        weights = np.array([qubit.adj(node) for node in self.nodes])
        return weights
    
    def log_matrix(self):
        return np.array([self.weights(node) for node in self.nodes])
    
    def matrix(self):
        return np.exp(-self.log_matrix())
    
    def prob_matrix(self):
        return np.abs(self.matrix())**2
    