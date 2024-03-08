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
    
    def change_basis(self, new_basis):
        '''
        change the basis of the qubit
        '''
        self.state = np.dot(np.conj(np.transpose(new_basis.states)), self.state)
        self.basis = new_basis
    
    def probs(self, basis = None):
        '''
        get the probabilities of measuring |0> and |1> in the given basis
        '''
        if basis is None:
            basis = self.basis
            
    def measure(self, basis = ['0', '1']):
        return 0