import numpy as np

class basis:
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

class qubit:
    '''
    qubit class
    '''
    def __init__(self, state = np.array([1,0]), basis = basis()):
        norm = np.linalg.norm(state)
        assert np.isclose(norm, 1), "State must be normalized"
        self.state = state
        self.basis = basis
        
    def __str__(self):
        return str(self.state)
    
    def __repr__(self):
        return str(self.state)

    def _copy(self):
        return qubit(state = self.state, basis = self.basis)
    
    def change_basis(self, new_basis):
        '''
        change the basis of the qubit
        '''
        self.state = np.dot(np.conj(np.transpose(new_basis.states)), self.state)
        self.basis = new_basis
    
    def probs(self, basis = None):
        '''
        get the probabilities of measuring a basis
        '''
        if basis is None:
            basis = self.basis
        copy_qubit = self._copy()
        copy_qubit.change_basis(basis)
        probs = np.abs(copy_qubit.state) ** 2
        return probs
            
    def measure(self, basis = None):
        '''
        measure the qubit in a basis
        '''
        if basis is None:
            basis = self.basis
        probs = self.probs(basis = basis)
        result = np.random.choice([0, 1], p = probs)
        if result == 0:
            self.state = [1, 0]
        else:
            self.state = [0, 1]
        return result