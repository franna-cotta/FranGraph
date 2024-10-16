from .graph import UndirectedGraph
import numpy as np
import scipy as sp
from itertools import pairwise, cycle

# Iterates pairwise through a list and loops back on itself for the final iteration
# e.g. closedPairwise('ABCDE') -> ['AB', 'BC', 'CD', 'DE', 'EA']
class closedPairwise():
    def __init__(self, iterable):
        self.iterator = pairwise(iterable)
        self.final = (iterable[-1], iterable[0])
        self.end = False
        
    def __iter__(self):
        return self
    
    def __next__(self):        
        if self.end:
            raise StopIteration
        
        else:
            try:
                a = next(self.iterator)
                return a

            except StopIteration:
                self.end = True
                return self.final[0], self.final[1]

# Iterates through ridges (lists of vertex indices that are colinear, coplanar, co-3-hyperplanar, etc.) and returns edges (pairs of indices)
# e.g. iterateRidges([[1, 2], [3, 4, 5]]) -> [[1,2], [3,4], [4,5], [5,6]]
def iterateRidges(ridges):
    for ridge in ridges:
        if -1 in ridge:
            continue
            
        elif len(ridge) == 2:
            yield ridge
            
        else:
            for edge in closedPairwise(ridge):
                yield edge            

# Extends the SciPy Voronoi class to add periodic boundary conditions by taking the input points, capturing those adjacent to the faces of the bounding hypercube, and duplicating them on the opposite face
class periodicVoronoi(sp.spatial.Voronoi):
    def __init__(self, points, 
                 bounds = None, # Specified as a pair of vectors denoting the lower-left and upper-right corners of the bounding region or equivalent
                 L = None, # Width of the buffer region around the border hypercube within which points are considered 'border points. Specfied as a float, calculated from the bounds if not given.
                 periodicIn = None, # Specifies which dimensions the Voronoi diagram must be periodic in
                 furthest_site = False, incremental = False, qhull_options = None):
        
        dim = len(points[0])
        points = np.array(points)
        
        if not bounds:
            bounds = (((-1.0,)*dim), ((1.0,)*dim))
        else:
            assert dim == len(bounds[0]), "The bounding box must have the same dimensions as the points"
            
        if not L:
            # Get the longest side of the simulation region
            L = max([bounds[1][i] - bounds[0][i] for i in range(dim)]) / 2.0
        
        unitVectors = np.eye(dim)
        
        if not periodicIn:
            periodicIn = ((True,)*dim)
        else:
            assert dim == len(periodicIn), "The bounding box must have the same dimensions as the points"
            
        # iterate through each dimension
        for i in range(dim):
            # Trim any points outside of the bounding region
            points = [p for p in points if p[i] >= bounds[0][i] and p[i] <= bounds[1][i]]
            
        # Iterate through each dimension again
        for i in range(dim):
            if periodicIn[i]:
                shift = bounds[1][i] - bounds[0][i]

                # Get the points within the buffer region
                # Shift them too
                leftDups = [p + shift * unitVectors[i] for p in points if p[i] <= bounds[0][i] + L]
                rightDups = [p - shift * unitVectors[i] for p in points if p[i] >= bounds[1][i] - L]

                # Add them to the existing points
                points = points + leftDups + rightDups
            
        super().__init__(points, furthest_site = furthest_site, incremental = incremental, qhull_options = qhull_options)   
        
# A power diagram is a weighted d-dimensional Voronoi diagram 
# We use an algorithm that generates a (d+1)-dimensional Voronoi diagram with the weights as the additional dimension
# Then projects the completed (d+1)-dimensional cells back down to a d-dimensional structure
class powerDiagram():
    def __init__(self, points, weights, bounds = None, periodicIn = None):
        if len(points) != len(weights):
            raise Exception("Number of points and weights must match")
        
        self._dim = len(points[0])
        self._points = np.array(points, dtype = 'float64')
        self._weights = np.array(weights, dtype = 'float64').reshape(-1,1)
        self._vertices = list()
        self._vertex_indices = set()
        self._edges = list()
        self._voronoi_edges = dict()
        self._vertex_map = dict()
        self._graph = None
        self._bounds = bounds
        self._periodicIn = periodicIn
        
        self.regenerate()
        
    @property
    def edges(self):
        return self._edges
    
    @property
    def vertices(self):
        return self._vertices
        
    def edge_exists(self, edge):
        return tuple(edge) in self._edges or tuple(edge[::-1]) in self._edges
    
    def intersects_plane(self, pts):
        wts = pts[:, -1]
        
        return not(np.all(wts < 0.0) or np.all(wts > 0.0))
    
    def totally_in_plane(self, pts):
        return np.all(pts[:,-1] == 0.0)
    
    def partially_in_plane(self, pts):
        return np.any(pts[:,-1] == 0.0)
    
    def in_bounds(self, pt):
        if self._bounds:
            return np.all(pt < self._bounds[1]) and np.all(pt > self._bounds[0])
        else:
            return True
        
    def dimensionality(self, pts):
        M = pts[:-1] - pts[-1]
        
        return np.linalg.matrix_rank(M)               
        
    def process_ridge(self, ridge):              
        # Collect the real-valued coordinates
        pts = self._voronoi.vertices[ridge]
        
        # First check the ridge lies neither entirely above or below the plane
        # If it does, then it clearly cannot intersect the plane
        if self.intersects_plane(pts):            
            # Calculate the dimensionality (rank of the matrix of differences)
            # This tells us the dimension of the subspace spanned by the ridge. E.g. is the ridge a line, a polygon, a polyhedra, a polytope, etc.
            dim = self.dimensionality(pts)
            
            if len(ridge) == 2 or dim == 1:
                # The ridge is a line segment, which means the intersection with the plane is a vertex
                # We can disregard this case since it will just created isolated points
                print(f"Vertex intersection. Dimensionality is {dim}, ridge is {ridge}")
                pass

            elif len(ridge) == 3 or dim == 2:
                # print("Line segment intersection")
                # The ridge is a polygon, which means the intersection with the plane is a line segment (or the polygon itself, if all vertices lie in the plane)
                
                # Check first whether all the points lie in the plane
                if self.totally_in_plane(pts):
                    print(f"Polygon {ridge} lies totally in the plane")
                    
                elif self.partially_in_plane(pts):
                    print(f"Polygon {ridge} has vertices lying in the plane")
                
                # If not, then we need to compute the intersections
                else:                    
                    # Create a list to track the new vertices formed from the intersections                    
                    new_verts = list()
                    new_edge = list()

                    # Iterate through each edge in the polygon
                    for i1, i2 in closedPairwise(ridge):
                    
                        if i1 == -1 or i2 == -1:
                            # If either of the points are a -1 we have to skip it
                            # Because these represent points outside the Voronoi diagram
                            continue
                    
                        # Check if the edge is already known as an intersection 
                        # i.e. a shared edge of another ridge already iterated
                        elif (i1, i2) in self._voronoi_edges:
                            # Simply append the known vertex to the new_edge
                            new_edge.append(self._voronoi_edges[(i1, i2)])

                        # Otherwise...
                        else:
                            # Get the real coordinates (a little redundant)
                            p1 = self._voronoi.vertices[i1]
                            p2 = self._voronoi.vertices[i2]

                            # Check if two points intersect the plane (are on opposite sides of the plane)
                            # Also check they're in-bounds
                            if ((p1[-1] > 0 and p2[-1] < 0) or (p1[-1] < 0 and p2[-1] > 0)):# and self.in_bounds(p1) and self.in_bounds(p2):
                                uv = np.array(p2) - np.array(p1) # Vector from p1 to p2

                                # The intersection occurs when (p1 + a * uv)[-1] = 0.0, we need to solve for a
                                # ==> p1[-1] + a * uv[-1] = 0.0
                                # ==> a = -p1[-1] / uv[-1]
                                intersection = p1 - (p1[-1] / uv[-1]) * uv

                                # Reduce the dimension of the intersection by one, treating it as a vertex in the plane
                                intersection = intersection[:-1]

                                # Append the new vertex to the list
                                new_verts.append(((i1, i2), intersection))
                                #print(f"Adding vertex intersection {(i1, i2)}")

                    # Verify there are exactly two points of intersection
                    # If we don't do this, issues with points at infinity can occur
                    if len(new_edge) + len(new_verts) == 2:
                        # Add any new points to the edge map
                        # self._voronoi_edges tracks the correspondence between edges in the voronoi complex and vertices in the intersected plane
                        # This way if the same voronoi edge intersection crops up, we don't duplicate the intersection vertex
                        for voronoi_edge, intersection_vertex in new_verts:
                            # Add the new intersection vertex to the list of vertices
                            self._vertices.append(intersection_vertex)

                            # Get the index of the newly-added point
                            intersection_vertex_index = len(self._vertices) - 1

                            # Append the new index to the edge
                            new_edge.append(intersection_vertex_index)

                            # Map the indices of the edges intersecting the plane in the voronoi complex
                            # to the index of the point of intersection in self._vertices
                            self._voronoi_edges[voronoi_edge] = intersection_vertex_index
                            self._voronoi_edges[voronoi_edge[::-1]] = intersection_vertex_index

                        new_edge = tuple(new_edge)

                        # Append the new edge to the list of edges
                        if new_edge not in self._edges:
                            self._edges.append(new_edge)

            elif len(ridge) == 4 or dim == 3:
                print(f"Polygonal intersection. Dimensionality is {dim}, ridge is {ridge}")
                # Polytopal ridge, the intersection with the plane is a bounded polygon
                # If we're dealing with a 2D power diagram, we can disregard this and reconstruct it with an UndirectedGraph clas
                new_verts = list()

                # TO DO but not high priority
                for i1, i2 in closedPairwise(ridge):
                    if i1 == -1 or i2 == -1:
                        # If either of the points are a -1 we have to skip it
                        # Because these represent points outside the Voronoi diagram
                        continue
                            
                        """uv = np.array(p2) - np.array(p1) # Unit vector along the direction between the two points
                        uv /= np.linalg.norm(uv)
                        
                        # Need to solve (p1 + a * uv)[-1] = 0.0 for a
                        # ==> p1[-1] + a * uv[-1] = 0.0
                        # ==> a = -p1[-1] / uv[-1]
                        intersection = np.array(p1) + (- p1[-1] / uv[-1]) * uv
                        
                        # Reduce the dimension of the intersection by one
                        intersection = intersection[:-1]
                        print(f"Intersection {intersection}")
                        
                        # Append the new vertex to the list, along with the new index
                        new_verts.append(intersection)
                        #new_indices[len(new_verts) - 1] = len(self.vertices) + len(new_verts) - 1
                        
                # Take the convex hull
                CH = sp.spatial.ConvexHull(new_verts)
                
                # Fetch the edges and recreate them with the new vertices
                new_edges = [(i + len(self._vertices), j + len(self._vertices)) for i, j in edge for edge in CH.simplices]
                
                self._vertices += new_verts
                self._edges += new_edges"""
                
            else:
                print("Polytopal intersection")
                # Higher-dimensional polytopal ridge, disregarded for now
                pass  
            
            #print(f"Found Total: {self.numFound}") 
        #else:
            #print("Does not intersect the plane")
        
        
    def regenerate(self, offset = 0.0):
        # Get the max weight
        C = max(self._weights)
        
        # Generate a Voronoi structure with the square root of the weights as per Observation 7 of Lévy
        if self._bounds:
            self._voronoi = periodicVoronoi(points = np.hstack((self._points, np.sqrt(C - self._weights) - offset)),
                                       bounds = self._bounds,
                                        periodicIn = self._periodicIn)
        else:
            self._voronoi = sp.spatial.Voronoi(points = np.hstack((self._points, np.sqrt(C - self._weights) - offset)))
            
        
        # To find the vertices, we need to trace through every ridge in the Voronoi diagram and
        # check if they intersect the hyperplance R^d
        for ridge in self._voronoi.ridge_vertices:
            if -1 not in ridge:
                self.process_ridge(ridge)  
                
    def draw(self):
        import matplotlib.pyplot as plt
        
        for e in self._edges:
            p1, p2 = self._vertices[e[0]], self._vertices[e[-1]]
                    
            plt.plot((p1[0], p2[0]), (p1[-1], p2[-1]), color = 'black')
            
        plt.figure(figsize=(16,16))
        plt.show()
        plt.clf()
        
        return plt

        

    
