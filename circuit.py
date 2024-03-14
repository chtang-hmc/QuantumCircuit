import numpy as np
from qubit import *
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.visualization import plot_histogram, circuit_drawer

class Gate:
    '''
    Quantum Gate
    '''
    def __init__(self, matrix: np.array, name: str = None):
        self.matrix = matrix
        self.dim = len(matrix)
        self.name = name
        
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name
    
    def __mul__(self, other):
        assert self.dim == other.dim, "Gates must be of the same dimension"
        return Gate(np.dot(self.matrix, other.matrix), name = f'{self.name} * {other.name}')
    
    def __pow__(self, n):
        return Gate(np.linalg.matrix_power(self.matrix, n), name = f'{self.name}^{n}')
    
    def __call__(self, qubit: Qubit):
        return Qubit(np.dot(self.matrix, qubit.state))
    
    def __getitem__(self, i):
        return self.matrix[i]
    
    def __setitem__(self, i, value):
        self.matrix[i] = value
        
    def __eq__(self, other):
        return np.allclose(self.matrix, other.matrix)
    
    def __ne__(self, other):
        return not np.allclose(self.matrix, other.matrix)
    
    def tensor_product(self, other):
        '''
        tensor product of two gates
        '''
        return Gate(np.kron(self.matrix, other.matrix), name = f'{self.name} x {other.name}')
    
    def dagger(self):
        '''
        Hermitian conjugate
        '''
        return Gate(np.conj(self.matrix).T)
    
    def adjoint(self):
        '''
        adjoint
        '''
        return Gate(np.conj(self.matrix).T)
    
    def is_unitary(self):
        '''
        check if gate is unitary
        '''
        return np.allclose(np.dot(self.matrix, np.conj(self.matrix).T), np.eye(self.dim))
    
    def is_hermitian(self):
        '''
        check if gate is hermitian
        '''
        return np.allclose(self.matrix, np.conj(self.matrix).T)

class QC:
    '''
    Quantum Circuit
    '''
    def __init__(self, dim: int):
        self.qubits = [Qubit() for _ in range(dim)]
        self.dim = dim
        self.gates = [[] for _ in range(self.dim)]
        
    def __str__(self):
        '''
        print circuit
        '''
        qiskit_circuit = self._to_qiskit()
        return str(circuit_drawer(qiskit_circuit, output='text'))
    
    def _to_qiskit(self):
        '''
        convert to qiskit circuit
        '''
        circuit = QuantumCircuit(self.dim, self.dim)
        for qubit in range(self.dim):
            for gate in self.gates[qubit]:
                if gate.name == 'H':
                    circuit.h(qubit)
                elif gate.name == 'X':
                    circuit.x(qubit)
                elif gate.name == 'Y':
                    circuit.y(qubit)
                elif gate.name == 'Z':
                    circuit.z(qubit)
                elif gate.name == 'I':
                    pass
                elif gate.name == 'S':
                    circuit.s(qubit)
                elif gate.name == 'T':
                    circuit.t(qubit)
                elif gate.name == 'S†':
                    circuit.sdg(qubit)
                elif gate.name == 'T†':
                    circuit.tdg(qubit)
                elif 'Rx' in gate.name:
                    circuit.rx(qubit, float(gate.name[3:-1]))
                elif 'Ry' in gate.name:
                    circuit.ry(qubit, float(gate.name[3:-1]))
                elif 'Rz' in gate.name:
                    circuit.rz(qubit, float(gate.name[3:-1]))
                else:
                    raise Exception(f"Gate {gate.name} not supported")
        return circuit
        
    def show(self):
        '''
        show circuit
        '''
        qiskit_circuit = self._to_qiskit()
        draw = qiskit_circuit.draw(output = 'mpl')
        return draw
    
    
    def _add_gate(self, gate: Gate, qubit: int):
        '''
        add a gate to the circuit
        '''
        assert gate.dim == 2, "Gate must be 2D"
        assert qubit < self.dim, "Qubit index out of range"
        for i in range(qubit):
            self.gates[i].append(Gate(np.eye(2)))
        self.gates[qubit].append(gate)
        
    def _apply_gates(self, qubit: int):
        '''
        apply gates to a qubit
        '''
        for gate in self.gates[qubit]:
            self.qubits[qubit] = gate(self.qubits[qubit])
    
    def _apply_entanglement(self, qubit1: int, qubit2: int):
        '''
        apply entanglement between two qubits
        '''
        self.qubits[qubit1].entangle(self.qubits[qubit2])
        
    def _apply_measurement(self, qubit: int):
        '''
        measure a qubit
        '''
        self.qubits[qubit].measure()
    
    def _apply_circuit(self):
        '''
        apply the circuit
        '''
        for qubit in range(self.dim):
            self._apply_gates(qubit)
            
    def h(self, qubit: int):
        '''
        apply Hadamard gate
        '''
        self._add_gate(Gate(np.array([[1, 1], [1, -1]])/np.sqrt(2), name = 'H'), qubit)
    
    def x(self, qubit: int):
        '''
        apply Pauli-X gate
        '''
        self._add_gate(Gate(np.array([[0, 1], [1, 0]]), name = 'X'), qubit)
        
    def y(self, qubit: int):
        '''
        apply Pauli-Y gate
        '''
        self._add_gate(Gate(np.array([[0, -1j], [1j, 0]]), name = 'Y'), qubit)
        
    def z(self, qubit: int):
        '''
        apply Pauli-Z gate
        '''
        self._add_gate(Gate(np.array([[1, 0], [0, -1]]), name = 'Z'), qubit)
        
    def s(self, qubit: int):
        '''
        apply S gate
        '''
        self._add_gate(Gate(np.array([[1, 0], [0, 1j]]), name = 'S'), qubit)
        
    def t(self, qubit: int):
        '''
        apply T gate
        '''
        self._add_gate(Gate(np.array([[1, 0], [0, np.exp(1j * np.pi/4)]]), name = 'T'), qubit)
        
    def sdg(self, qubit: int):
        '''
        apply S dagger gate
        '''
        self._add_gate(Gate(np.array([[1, 0], [0, -1j]]), name = 'S†'), qubit)
        
    def tdg(self, qubit: int):
        '''
        apply T dagger gate
        '''
        self._add_gate(Gate(np.array([[1, 0], [0, np.exp(-1j * np.pi/4)]]), name = 'T†'), qubit)
        
    def rx(self, qubit: int, theta: float):
        '''
        apply R_x() gate
        '''
        self._add_gate(Gate(np.array([[np.cos(theta/2), -1j*np.sin(theta/2)], [-1j*np.sin(theta/2), np.cos(theta/2)]]), name = f'Rx({theta})'), qubit)
        
    def ry(self, qubit: int, theta: float):
        '''
        apply R_y() gate
        '''
        self._add_gate(Gate(np.array([[np.cos(theta/2), -np.sin(theta/2)], [np.sin(theta/2), np.cos(theta/2)]]), name = f'Ry({theta})'), qubit)
        
    def rz(self, qubit: int, theta: float):
        '''
        apply R_z() gate
        '''
        self._add_gate(Gate(np.array([[np.exp(-1j*theta/2), 0], [0, np.exp(1j*theta/2)]]), name = f'Rz({theta})'), qubit)
        
    def id(self, qubit: int):
        '''
        apply identity gate
        '''
        self._add_gate(Gate(np.eye(2), name = 'I'), qubit)
        
    def cx(self, control: int, target: int):
        '''
        apply CNOT gate
        '''
        self.id(control)
        self._add_gate(Gate(np.array([[0, 1], [1, 0]]), name = 'X'), target)
        self.qubits[control].entangle(self.qubits[target])
        
    def cz(self, control: int, target: int):
        '''
        apply CZ gate
        '''
        self.id(control)
        self._add_gate(Gate(np.array([[1, 0], [0, -1]]), name = 'Z'), target)
        self.qubits[control].entangle(self.qubits[target])
        
    def swap(self, qubit1: int, qubit2: int):
        '''
        apply SWAP gate
        '''
        self.cx(qubit1, qubit2)
        self.cx(qubit2, qubit1)
        self.cx(qubit1, qubit2)
        
    def controlled(self, gate: Gate, control: int, target: int):
        '''
        apply controlled gate
        '''
        self.id(control)
        self._add_gate(gate, target)
        self.qubits[control].entangle(self.qubits[target])
    
    def measure(self, qubit: int):
        '''
        measure a qubit
        '''
        self._apply_measurement(qubit)
        
    def get_final_states(self):
        self._apply_circuit()
        return self.qubits
        
    def run(self, shots = 1024):
        '''
        run the circuit without using qiskit
        '''
        self._apply_circuit()
        results = []
        for qubit in self.qubits:
            results.append(qubit.measure())
        return results
    
    def multi_run(self, shots = 1024):
        '''
        run the circuit without using qiskit
        '''
        for shot in range(shots):
            self._apply_circuit()
            results = []
            for qubit in self.qubits:
                results.append(qubit.measure())
            yield results
            
    def _single_equiv_gate(self, qubit: int):
        '''
        convert single wire to matrix
        '''
        equiv_gate = Gate(np.eye(2), name = 'I')
        for gate in self.gates[qubit]:
            equiv_gate = equiv_gate * gate
        return equiv_gate
    
    def _multi_equiv_gate(self, qubits: list):
        '''
        convert multi wire to matrix
        '''
        equiv_gate = self._single_equiv_gate(qubits[0])
        for qubit in qubits[1:]:
            equiv_gate = equiv_gate.tensor_product(self._single_equiv_gate(qubit))
        return equiv_gate
    
    def equiv_gate(self):
        '''
        get equivalent gate
        '''
        return self._multi_equiv_gate(list(range(self.dim)))