import numpy as np
import numba as nb
from itertools import product
from sympy import divisors

def rotationMatrix2D(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c, -s), (s, c)))

reflectionMatrix2D = np.array(((1,0),(0,-1)))

# Given an input matrix M and a list of matrices L
# Goes through each element l in L and checks if M is identical
# If there is an identical matrix, returns False. Otherwise True.
def uniqueMatrixInList(M, L):
    for l in L:
        if np.allclose(M, l):
            return False
    return True
        
    
class symmetrify:
    def __init__(self, coords, symmetry):  
        self._coords = np.array(coords)
        self._orbits = list()
        self._symmetry = symmetry
        
        for i, c in enumerate(self._coords):
            orbit = self._symmetry.generateOrbit(np.array(c), reduced = False)
            
            orbitByIndex = dict()
            
            for k, v in orbit.items():
                orbitByIndex[k] = self.getIndexByCoord(v)
            
            self._orbits.append(orbitByIndex)

    def __getitem__(self, key):
        return self.orbit_of(key)        

    @property
    def coords(self):
        return self._coords
    
    @property
    def orbits(self):
        return self._orbits

    @property
    def symmetry(self):
        return self._symmetry

    def orbit_of(self, key):
        return [i for i in self._orbits[key].values() if i]
            
    # Simple utility function to get a node given a set of coordinates, if possible
    def getIndexByCoord(self, coords, omit = list()):
        for i, v in enumerate(self._coords):            
            if i in omit:
                continue
                
            if np.allclose(coords, v, atol = 1e-2):
                return i
        return None
        

# For now, symmetry groups are defined in terms of matrices and the enumeration of all their products
class symmetryGroup:

    def __init__(self, generators = list(), named = '', dim = 2):
        if named != '':
            # Named groups
            
            # Cyclic group
            if named[0] == 'C' and named[1:].isdigit(): 
                if dim == 2:
                    theta = np.radians(360 / int(named[1:]))

                    generators = {'r' : rotationMatrix2D(theta)}
                
            # Dihedral group
            if named[0] == 'D' and named[1:].isdigit():
                if int(named[1:]) % 2 == 0:
                    if dim == 2:
                        theta = np.radians(360 / (int(named[1:])/2))

                        generators = {'r' : rotationMatrix2D(theta), 's' : reflectionMatrix2D}
                else:
                    print("Dihedral group must be of even order")
                    
            # Full tetrahedral group
            # https://arxiv.org/pdf/1910.07143.pdf
            if named == 'T' and dim == 3:
                generators = {'a' : np.array(((1,0,0),(0,-1,0),(0,0,-1))),
                              'b' : np.array(((0,-1,0),(0,0,1),(-1,0,0))),
                              'c' : np.array(((1,0,0),(0,0,-1),(0,-1,0)))}               
        
        self.generators = generators
        self.elements = {'e' : np.eye(dim)}
        self.elements.update(generators)
        
        # Start by finding the powers of the base elements
        for k, g in generators.items():            
            O = 2
            
            while O < 100: # Arbitrary break point
                M = np.linalg.matrix_power(g, O)
                
                # If we have formed the identity matrix, then we've found the order of the element and can break the loop
                if np.allclose(M, np.eye(dim)):
                    break
                    
                # Otherwise we've found a new, unique element, so don't break the loop
                else:
                    key = k*O
                    self.elements[key] = M
                    #self.elements.append(M)
                    
                O += 1
                
        # Then find all the multiplicative combinations that are unique
        for p in product(self.elements.copy().items(), repeat = 2):
            k = p[0][0] + p[1][0]
            M = p[0][1] @ p[1][1]
            
            if uniqueMatrixInList(M, self.elements.values()):
                #self.elements.append(M)
                self.elements[k]= M
                
        self.order = len(self.elements)
        self.suborders = divisors(self.order)[1:-1]
                
    def generateOrbit(self, p, reduced = True):
        if reduced:
            orbit = [p,]
        else:
            orbit = {'e' : p}
        
        for k, g in self.elements.items():
            q = g @ p
            
            if reduced:
                if uniqueMatrixInList(q, orbit.values()):
                    orbit.append(q)      
            else:
                orbit[k] = q
                
        return orbit
            
        