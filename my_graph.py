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
            
    def _measure_without_change(self, qubit):
        if isinstance(qubit, int):
            qubit = self.nodes[qubit]
        prob = self.root.prob(qubit)
        return np.random.choice([qubit, self.other(qubit)], p=[prob, 1-prob])
    
    def measure_without_change(self, qubit):
        return self._measure_without_change(qubit).state
    
    def _measure(self, qubit):
        if isinstance(qubit, int):
            qubit = self.nodes[qubit]
        result = self._measure_without_change(qubit)
        for edge in self.edges:
            if edge.v == qubit:
                edge.set_weight(0)
            if edge.v == self.other(qubit) and edge.u != self.other(qubit):
                self.delete_edge(edge = edge)
        return result
    
    def measure(self, qubit):
        return self._measure(qubit).state
    
    def _measure_and_change_root(self, qubit):
        if isinstance(qubit, int):
            qubit = self.nodes[qubit]
        result = self._measure(qubit)
        self.root = result
        return result
    
    def measure_and_change_root(self, qubit):
        return self._measure_and_change_root(qubit).state
    
    def _multimeasure(self, qubits):
        results = []
        for qubit in qubits:
            result = self._measure_and_change_root(qubit)
            results.append(result)
        return results
    
    def multimeasure(self, qubits):
        return [qubit.state for qubit in self._multimeasure(qubits)]
    
    def weights(self, qubit):
        if isinstance(qubit, int):
            qubit = self.nodes[qubit]
        weights = np.array([qubit.adj(node) for node in self.nodes])
        return weights
    
    def log_matrix(self):
        return np.array([self.weights(node) for node in self.nodes])
    
    def matrix(self):
        return np.exp(-self.log_matrix())
    
    def prob_matrix(self):
        return np.abs(self.matrix())**2
    
    def plot(self):
        plt.matshow(self.prob_matrix())
        plt.show()
        
    def _single_qubit_gate(self, qubit, gate):
        if isinstance(qubit, int):
            qubit = self.nodes[qubit]
        # for every node, check if there is an edge from qubit to node
        a,b,c,d = gate[0][0], gate[0][1], gate[1][0], gate[1][1]
        for node in self.nodes:
            if node != qubit:
                x, y = node.adj(qubit), node.adj(self.other(qubit))
                if node.neighbor(qubit):
                    node.set_weight(qubit, -np.log(np.exp(-x)*a + np.exp(-y)*b))
                else:
                    self.add_edge(node, qubit, -np.log(np.exp(-x)*a + np.exp(-y)*b))
                if node.neighbor(self.other(qubit)):
                    node.set_weight(self.other(qubit), -np.log(np.exp(-x)*c + np.exp(-y)*d))
                else:
                    self.add_edge(node, self.other(qubit), -np.log(np.exp(-x)*c + np.exp(-y)*d))
                    
    def X(self, qubit):
        self._single_qubit_gate(qubit, np.array([[0,1],[1,0]]))
        
    def Y(self, qubit):
        self._single_qubit_gate(qubit, np.array([[0,-1j],[1j,0]]))
        
    def Z(self, qubit):
        self._single_qubit_gate(qubit, np.array([[1,0],[0,-1]]))
        
    def H(self, qubit):
        self._single_qubit_gate(qubit, 1/np.sqrt(2)*np.array([[1,1],[1,-1]]))
        
    def create_equal_superposition(self):
        for i in range(1, self.size+1):
            self.H(i)
        
    def T(self, qubit):
        self._single_qubit_gate(qubit, np.array([[1,0],[0,np.exp(1j*np.pi/4)]]))
    