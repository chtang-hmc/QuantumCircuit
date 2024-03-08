import numpy as np

class basis:
    def __init__(self, states = np.eye(2), normalized = True):
        '''
        states (matrix): state in |0>, |1> basis
        '''
        assert len(states) == 2, "Basis must be 2D"
        assert len(states[i] for i in range(2)) == 2, "Basis must be 2D"
        assert np.isclose(np.dot(np.conj(states[0]), states[1]), 0), "Basis states must be orthogonal"
        if normalized:
            assert np.isclose(np.linalg.det(states), 1), "Basis states must be normalized"
        else:
            assert np.linalg.det(states) != 0, "Basis states must be linearly independent"
            # normalize states
        
        self.states = states
        self.dim = 2
        
    def __str__(self):
        return str(self.states)
    
    def __repr__(self):
        return str(self.states)

class qubit:
    def __init__(self, state, basis = ['0', '1']):
        norm = np.linalg.norm(state)
        assert np,isclose(norm, 1), "State must be normalized"
        self.state = state
        self.basis = basis
        
    def __str__(self):
        return str(self.state)
    
    def __repr__(self):
        return str(self.state)
    
    def probs(self, basis = None):
        if basis is None:
            basis = self.basis
            
        return np.abs(self.state)**2
    
    def measure(self, basis = ['0', '1']):
        