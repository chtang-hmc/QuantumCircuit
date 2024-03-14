import numpy as np

def get_orthogonal_basis(v):
    '''
    get an orthogonal basis from a vector
    '''
    v = v/np.linalg.norm(v)
    if np.isclose(v[0], 0):
        if np.isclose(v[1], 0):
            return np.array([[1, 0], [0, 1]])
        else:
            return np.array([[0, 1], [-1, 0]])
    else:
        return np.array([v, [-v[1], v[0]]])

class Basis:
    '''
    basis class
    '''
    def __init__(self, special: str = None, states = None, normalize = False):
        '''
        states (matrix): state in |0>, |1> basis
        '''
        if states is None:
            if special is not None:
                if special == '+-':
                    states = np.array([[1, 1], [1, -1]])/np.sqrt(2)
                elif special == '01':
                    states = np.array([[1, 0], [0, 1]])
                elif special == '10':
                    states = np.array([[0, 1], [1, 0]])
                elif special == 'RL':
                    states = np.array([[1, 1j], [1, -1j]])/np.sqrt(2)
            else:
                states = np.eye(2)
                
        assert len(states) == 2, "Basis must be 2D"
        for state in states:
            assert len(state) == 2, "Basis must be 2D"
        assert np.isclose(np.dot(np.conj(states[0]), states[1]), 0), "Basis states must be orthogonal"
        if normalize:
            states = np.array([states[0]/np.linalg.norm(states[0]), states[1]/np.linalg.norm(states[1])])
        else:
            assert np.isclose(np.linalg.norm(states[0]), 1) and np.isclose(np.linalg.norm(states[1]), 1), "Basis states must be normalized"
        
        self.states = states
        self.dim = 2
        
    def __str__(self):
        return str(self.states)
    
    def __repr__(self):
        return str(self.states)

class Qubit:
    '''
    qubit class
    '''
    def __init__(self, state: np.array = np.array([1,0]), basis: Basis = Basis(), entangled: bool = False, other: list = []):
        norm = np.linalg.norm(state)
        assert np.isclose(norm, 1), "State must be normalized"
        self.state = state
        self.basis = basis
        self.entangled = entangled
        self.other = other
        
    def __str__(self):
        return f'Qubit in state {self.state} in basis {self.basis}' + (f' entangled with {self.other}' if self.entangled else '')
    
    def __repr__(self):
        return f'Qubit in state {self.state} in basis {self.basis}' + (f' entangled with {self.other}' if self.entangled else '')

    def _copy(self):
        return Qubit(state = self.state, basis = self.basis)
    
    def change_basis(self, new_basis):
        '''
        change the basis of the qubit
        '''
        self.state = np.dot(np.conj(np.transpose(new_basis.states)), self.state)
        self.basis = new_basis
    
    def probs(self, basis = None, seed = 42):
        '''
        get the probabilities of measuring a basis
        '''
        np.random.seed(seed)
        if self.entangled:
            pass
        
        else:
            if basis is None:
                basis = self.basis
            copy_qubit = self._copy()
            copy_qubit.change_basis(basis)
            probs = np.abs(copy_qubit.state) ** 2
            return probs
            
    def measure(self, basis = None, seed = 42):
        '''
        measure and collapse the qubit in a basis
        '''
        np.random.seed(seed)
        if self.entangled:
            pass
        
        else:
            if basis is None:
                basis = self.basis
            probs = self.probs(basis = basis, seed = seed)
            result = np.random.choice([0, 1], p = probs)
            
            original_basis = self.basis
            self.state = [0, 0]
            self.state[result] = 1
            self.basis = basis
            self.change_basis(original_basis)
            return result

    def entangle(self, other):
        '''
        entangle two qubits
        '''
        self.entangled = True
        other.entangled = True
        self.other.append(other)
        other.other.append(self)