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
                self.add_edge(self.nodes[i+1+size], self.nodes[j+1+size], 0)
                self.add_edge(self.nodes[j+1+size], self.nodes[i+1+size], 0)
        for node in self.nodes[1:]:
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
        plt.colorbar()
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
        
    def R_z(self, qubit, theta):
        self._single_qubit_gate(qubit, np.array([[1,0],[0,np.exp(1j*theta)]]))
        
    def R_y(self, qubit, theta):
        self._single_qubit_gate(qubit, np.array([[np.cos(theta/2),-np.sin(theta/2)],[np.sin(theta/2),np.cos(theta/2)]]))
        
    def R_x(self, qubit, theta):
        self._single_qubit_gate(qubit, np.array([[np.cos(theta/2),-1j*np.sin(theta/2)],[-1j*np.sin(theta/2),np.cos(theta/2)]]))
        
    def _controlled_gate(self, control, target, gate):
        if isinstance(control, int):
            control = self.nodes[control]
        if isinstance(target, int):
            target = self.nodes[target]
        a,b,c,d = gate[0][0], gate[0][1], gate[1][0], gate[1][1]
        x, y = control.adj(target), control.adj(self.other(target))
        # find edges from control to target and target.other
        edge1 = self.get_edge(control, target)
        edge2 = self.get_edge(control, self.other(target))
        if edge1:
            edge1.set_weight(-np.log(np.exp(-x)*a + np.exp(-y)*b))
        else:
            self.add_edge(control, target, -np.log(np.exp(-x)*a + np.exp(-y)*b))
        if edge2:
            edge1.set_weight( -np.log(np.exp(-x)*c + np.exp(-y)*d))
        else:
            self.add_edge(control, self.other(target), -np.log(np.exp(-x)*c + np.exp(-y)*d))
            
    def CNOT(self, control, target):
        self._controlled_gate(control, target, np.array([[1,0],[0,1]]))
        
    def CCNOT(self, control1, control2, target):
        self.CNOT(control2, target)
        self.CNOT(control1, target)
        self.CNOT(control2, target)
        
    def Toffoli(self, control1, control2, target):
        self.CCNOT(control1, control2, target)
        
    def swap(self, qubit1, qubit2):
        self.CNOT(qubit1, qubit2)
        self.CNOT(qubit2, qubit1)
        self.CNOT(qubit1, qubit2)
        
    def CX(self, control, target):
        self.CNOT(control, target)
        
    def CZ(self, control, target):
        self._controlled_gate(control, target, np.array([[1,0],[0,-1]]))
        
    def CS(self, control, target):
        self._controlled_gate(control, target, np.array([[1,0],[0,1j]]))
        
    def CT(self, control, target):
        self._controlled_gate(control, target, np.array([[1,0],[0,np.exp(1j*np.pi/4)]]))
        
    def CR_z(self, control, target, theta):
        self._controlled_gate(control, target, np.array([[1,0],[0,np.exp(1j*theta)]]))
        
    def CR_y(self, control, target, theta):
        self._controlled_gate(control, target, np.array([[np.cos(theta/2),-np.sin(theta/2)],[np.sin(theta/2),np.cos(theta/2)]]))
        
    def CR_x(self, control, target, theta):
        self._controlled_gate(control, target, np.array([[np.cos(theta/2),-1j*np.sin(theta/2)],[-1j*np.sin(theta/2),np.cos(theta/2)]]))
        
    def is_equal(self, other):
        return np.allclose(self.prob_matrix(), other.prob_matrix())
    
    def __eq__(self, other):
        return self.is_equal(other)
    
    def __ne__(self, other):
        return not self.is_equal(other)