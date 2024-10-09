import numpy as np
import itertools
from more_itertools import zip_offset
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import scipy as sp
import matplotlib.colors
#import rigidpy as rp
import shapely as shp
from shapely.ops import unary_union
import gmsh as gm
import pyelmer as el
import os
import csv
from .symmetry import symmetryGroup
import time
from math import isclose

# Aperiodic Analyser
# This class constructs a tiling as an undirected graph, but taking in nodal coordinates and pairs of indices which are connected via an edge
# It categorizes data into four subtypes hierarchical: vertex (or nodal), edges, faces, and global
# Each subtype contains links to other elements connected to it, or which it forms a part of
# As well as utility functions for performing calculations relevant to a certain data type (e.g. face area)
class UndirectedGraph:
    import matplotlib.pyplot as plt
    
    # Takes a set of nodes, specified as real coordinates
    def __init__(self, node_coordinates, edge_indices, 
                 normalized = True, 
                 face_index_loops = None, 
                 dual = None, 
                 symmetry = '',
                 remove_asymmetric = False):
        self.initialized = False
        
        # Dimension of the space we are operating in        
        self.dim = len(node_coordinates[0])
        
        if self.dim < 2 or self.dim > 3:
            raise Exception("Currently only dimensions 2 and 3 are supported.")
        
        self.bounds = None
        
        self.global_rotation = np.eye(self.dim)
        self.global_scale = np.eye(self.dim)
        self.global_translation = np.zeros_like(node_coordinates[0])
        self._centroid0 = sum(np.array(n) for n in node_coordinates)/len(node_coordinates) # Initial centroid value
        
        # Define all the nodes as follows
        # index : Incremented sequentially with each new node
        # coords : Contains the (real) coordinations of the node in Euclidean space
        # neighbours : Finds all edges that contain the index, i, and appends the non-i index, j
        # graph : The parent graph (this object)
        t0 = time.time()
        print(f"Initializing nodes...")
        self.nodes = [Node(index = i,
                           coords = node_coordinates[i],
                           neighbours = [[j for j in ei if j != i][0] for ei in edge_indices if i in ei],
                           graph = self) for i in range(len(node_coordinates))]   
        print(f"Nodes initialized, took {time.time()-t0:2f} seconds") 
             
        self.bbox = list()
        for i in range(self.dim):
            minima = min([n.coords[i] for n in self.nodes])
            maxima = max([n.coords[i] for n in self.nodes])
            
            self.bbox.append((minima, maxima))
                
                                   
        # Initialize the symmetry group and generate the orbits
        self.symmetry = None
        self.orbits = set()
        
        if symmetry != '':
            t0 = time.time()
            print(f"Initializing {symmetry} symmetry group...")
            self.symmetry = symmetryGroup(named = symmetry)
            print(f"Symmetry group initialized, took {time.time()-t0:2f} seconds")
            t0 = time.time()
            
            print(f"Initializing orbits...")
            
            # Store the node indices we have already found the orbits for
            found_orbits = list()
            
            for n in self.nodes:
                # If the orbit of the node has already been found, skip it
                if n.index in found_orbits:
                    continue

                # Otherwise...
                else:
                    # Generate the orbit of n, as coordinates
                    n_orbit_coords = self.symmetry.generateOrbit(n.coords)
                    
                    # Create an empty list to store the orbit of n, as node index references
                    n_orbit = list()
                    
                    # Iterate through the coordinates of the points in the orbit                    
                    for coords in n_orbit_coords:
                        
                        # Get the node corresponding to the coordinates (if it exists)
                        # Omit any already-found orbits
                        qt_node = self.getNodeByCoord(coords, omit = found_orbits)
                        
                        # If it does exist, append its index to the orbit on n
                        if qt_node != None:
                            n_orbit.append(qt_node.index)
                        
                    # It is possible that two different transformations in the group
                    # may take a point to the same point in its orbit
                    # so we need to filter out only the unique indices
                    n_orbit = frozenset(n_orbit)
                    
                    # Add the complete orbit to the global list of all orbits
                    self.orbits.add(n_orbit)
    
                    # Add the nodes to the list of orbits found
                    found_orbits += list(n_orbit)
                    
                    # For each node in the orbit, set the orbit
                    for m in n_orbit:
                        self[m].orbit = n_orbit
            
            print(f"Orbits initialized, took {time.time()-t0:2f} seconds")
            
        # Remove nodes which lack total symmetry
        if remove_asymmetric:
            t0 = time.time()
            print(f"Removing asymmetric nodes...")
            # Collect the asymmetric nodes
            print(list(self.orbits)[0])
            asymmetric_nodes = [orbit for orbit in self.orbits in len(orbit) < self.symmetry.order] 
            print(asymmetric_nodes)
            #[n for n in orbit for orbit in self.orbits if (len(orbit) < self.symmetry.order)]
            
            # Delete edges connected to asymmetric nodes
            edge_indices = [edge for edge in edge_indices if any(edge in asymmetric_nodes)]
            
            # Regenerate the nodes
            self.nodes = [Node(index = i,
                           coords = node_coordinates[i],
                           neighbours = [[j for j in ei if j != i][0] for ei in edge_indices if i in ei],
                           graph = self) for i in range(len(node_coordinates)) if i not in asymmetric_nodes]  
            
            print(f"Asymmetric nodes removed, took {time.time()-t0:2f} seconds")  
        
        # Generate the edges
        # i : index of first vertex
        # j : index of second vertex
        # graph : The parent graph (this object)
        t0 = time.time()
        print(f"Initializing edges...")
        self.edge_indices = [tuple(e) for e in edge_indices]
        self.edges = [Edge(i = e[0], j = e[1], graph = self) for e in edge_indices]
        self.edge_count = len(self.edges)
        self.uniform_edge_thickness = 0.0
        print(f"Edges initialized, took {time.time()-t0:2f} seconds")  
        
        # Generate the convex hull & delaunay triangulation
        t0 = time.time()
        print(f"Generating associated structures...")
        self.convex_hull = sp.spatial.ConvexHull([n.coords for n in self.nodes])
        self.delaunay = sp.spatial.Delaunay([n.coords for n in self.nodes])
        self.voronoi = sp.spatial.Voronoi([n.coords for n in self.nodes], qhull_options = "Qbb Qz")
        print(f"Associated structures generated, took {time.time()-t0:2f} seconds")  
        
        # Generate and collect the faces
        t0 = time.time()
        print(f"Initializing faces...")
        self.faces = [Face(vertex_loop = VL, edge_loop = EL, graph = self) for VL, EL in zip(*self.generate_faces())]
        
        # The 'exterior' is detected as one giant face, which we term the border face
        # Detect the border face and pop it
        # Crude method that assumes the face with the most vertices is the border
        borderFace = max(self.faces, key = lambda f : len(f.nodes))
        
        # Just remove it for now
        self.faces.remove(borderFace)
        print(f"Faces initialized, took {time.time()-t0:2f} seconds")  
        
        
        t0 = time.time()
        print(f"Initializing connectivity...")
        
        # Track boundary nodes and edges separately
        self.boundary_nodes = list()
        self.boundary_edges = list()
        
        # Retroactively connect the edges to their associaed faces
        for e in self.edges:
            e.faces = [f for f in self.faces if e in f.edges]
            
            if len(e.faces) == 1:
                e.boundary = True
                self.boundary_edges.append(e)
        
        # Retroactively connect the nodes to their associated edges and faces
        for n in self.nodes:
            edge_idx = [(n.index, ni) for ni in n.neighbours]
            
            for eidx in edge_idx:
                edge = self[eidx]
                
                if isinstance(edge, ReversedEdge):
                    edge = edge.flipped
                    
                n.edges.append(edge)
                
                # If the node is connected to two or more boundary edges, we denote it a boundary node
                # May not be technically correct
                if sum([int(e.boundary) for e in n.edges]) >= 2:
                    n.boundary = True
                    self.boundary_nodes.append(n)
                
            n.faces = [ef for ef in e.faces for e in n.edges]
            
        print(f"Connectivity initialized, took {time.time()-t0:2f} seconds")  
        
        # Used for directional data analysis and histograms
        self.num_bins = 360
        self.normalized = normalized
        
        self._EOD = None
        
        if dual:
            t0 = time.time()
            print(f"Initializing dual...")
            
            if isinstance(dual, str):
                if dual == 'voronoi':  
                    nc = self.voronoi.vertices
                    ei = list()

                    # Iterate through the voronoi regions
                    for r in self.voronoi.regions:
                        # Ignore edge or empty tiles
                        if (-1 in r) or (len(r) == 0):
                            continue

                        # If we have a non-edge tile
                        else:
                            # Loop the first element back on itself
                            r = r + [r[0]]

                            # Extract the edge indices as tuple
                            edges = [(r[i], r[i+1]) for i in range(len(r) - 1)]

                            # For each extracted edge
                            for e in edges:
                                # Check if it's already in the list of edge indices (ei)
                                if e not in ei and e[::-1] not in ei:
                                    # Add if not
                                    ei.append(e)

                    # Construct the dual
                    self.dual = UndirectedGraph(node_coordinates = nc,
                                                edge_indices = ei,
                                                normalized = normalized, 
                                                face_index_loops = None, # I forgot what this parameter is even for ngl
                                                dual = self,
                                                symmetry = symmetry)
                    
                elif dual == 'delaunay':
                    nc = self.delaunay.points
                    print(f"Number of points: {len(nc)}")
                    ei = list()

                    # Iterate through the voronoi regions
                    for r in self.delaunay.simplices:
                        # Loop the first element back on itself
                        r = r + [r[0]]

                        # Extract the edge indices as tuple
                        edges = [(r[i], r[i+1]) for i in range(len(r) - 1)]

                        # For each extracted edge
                        for e in edges:
                            # Check if it's already in the list of edge indices (ei)
                            if e not in ei and e[::-1] not in ei:
                                # Add if not
                                ei.append(e)

                    # Construct the dual (delaunay)
                    self.dual = UndirectedGraph(node_coordinates = nc,
                                                edge_indices = ei,
                                                normalized = normalized, 
                                                face_index_loops = None, # I forgot what this parameter is even for ngl
                                                dual = self,
                                                symmetry = symmetry)
                
            elif isinstance(dual, (tuple, list)) and (len(dual) == 2 or len(dual) == 3):
                if len(dual) == 3:
                    FL = dual[2]
                else:
                    FL = None
                
                self.dual = UndirectedGraph(node_coordinates = dual[0],
                                            edge_indices = dual[1],
                                            normalized = normalized, 
                                            face_index_loops = FL, 
                                            dual = self,
                                            symmetry = symmetry)                
            else:
                self.dual = dual
                
            print(f"Dual initialized, took {time.time()-t0:2f} seconds")  
            
        self.initialized = True
        
            
    def __getitem__(self, idx): 
        if isinstance(idx, int):
            if idx <= len(self.nodes):
                return self.nodes[idx]
            else:
                raise IndexError(idx)
            
        elif isinstance(idx, (tuple, list, set)):
            # Check all the entries are integers
            int_check = all([isinstance(i, int) for i in idx])
            val_check = False
            
            # Check all the integers are in the correct bounds
            if int_check:
                val_check = all([i <= len(self.nodes) for i in idx])
            
            # If both are satisfied, we proceed
            if int_check and val_check:
                
                # Edge case (literally)
                if len(idx) == 2:

                    if idx in self.edge_indices:
                        return self.edges[self.edge_indices.index(idx)]

                    elif idx[::-1] in self.edge_indices:
                        return ReversedEdge(i = idx[1], j = idx[0], graph = self) #, thickness = self.edges[self.edge_indices.index(idx)].thickness)
                        #return self.edges[self.edge_indices.index(idx[::-1])]      

                    else:
                        return Edge(i = idx[0], j = idx[1], graph = self, virtual = True)
                
                # Face case
                else:
                    pass
                
            else:
                raise IndexError(f"The entries of {idx} are not valid indexing types.")
        
        else:
            raise IndexError(f"{idx} is not a valid indexing type.")
                      
        
    def __repr__(self):
        out = ''
        
        for v in self.nodes:
            out += f"{str(v)},\n"
            
        return out
    
    def generate_faces(self, loop_limit = 1000):  
        print("Beginning face generation...")          
        
        # Final list of faces to be returned
        face_list = list()
        
        # Generate the faces
        # THIS CODE ASSUMES ALL FACES ARE CONVEX POLYGONS
        face_vertex_sets = list()
        face_vertex_loops = list()
        face_edge_loops = list()
        
        # Track which edges have been included in a face
        edge_face_counter = {ei : 0 for ei in self.edge_indices}

        edge_index_iter = itertools.cycle(self.edge_indices)
        
        num_edges = len(self.edge_indices)
        
        current_edge_index = next(edge_index_iter)
            
        loop_counter = 0
            
        # Iterate through all the edges until each has been assigned to exactly two faces
        while sum(edge_face_counter.values()) < 2 * num_edges:
            loop_counter += 1
            
            if loop_counter > loop_limit:
                print("Reached loop limit whilst generating faces.")
                break
            
            # If we encounter an edge that has already been assigned to two faces, skip it                
            while edge_face_counter[current_edge_index] >= 2:
                current_edge_index = next(edge_index_iter)
            
            # Store the index of the starting edge
            initial_edge_index = current_edge_index
            
            # Set the current edge
            current_edge = self[current_edge_index]

            # Get the indices of the first two vertices of the loop so we can track them. Equivalently, the vertices of the edge.
            vertex_loop = list(current_edge_index)

            # Form a loop of all the found edges so we can track those too
            edge_loop = [current_edge,]

            # We terminate when we reach the first node and complete the cycle
            terminal_node_index = vertex_loop[0]

            while True:
                # Get the final two entries in the vertex loop and store those
                prev_node_index = vertex_loop[-2]
                current_node_index = vertex_loop[-1]

                # Break the loop if we have reached the terminal index
                # This is the only way to break the loop
                if current_node_index == terminal_node_index:
                    break

                # Get the possible next nodes from the neighbours of the edge which include the current node
                possible_next_edges = [self[ni] for ni in (current_edge.forward_neighbours + current_edge.backward_neighbours) if current_node_index in ni]
                    
                if len(possible_next_edges) == 0:
                    raise Exception(f"Degenerate edge {current_edge}, no valid neighbours found!")
            
                # Get the (unsigned) angle of the current edge, making sure we are taking it from the perspective of the current node
                current_edge_angle = current_edge.angle_wrt_node(current_node_index, unsigned = True)
                
                # Define a quick lambda to simplify the next step
                angle_diff = lambda e : current_edge_angle - e.angle_wrt_node(current_node_index, unsigned = True)
                
                next_edge = max(possible_next_edges, key = lambda e : angle_diff(e) if angle_diff(e) >= 0 else 2*np.pi + angle_diff(e))
                        
                # Flip the next edge if necessary
                if isinstance(next_edge, ReversedEdge):
                    next_edge = next_edge.flipped
            
                # Get the index of the next node in the chain by picking the index of the next edge that is not the current node
                next_node_index = next_edge.indices[0] if next_edge.indices[1] == current_node_index else next_edge.indices[1] 

                # Append these to the vertex and edge loop
                vertex_loop.append(next_node_index)
                edge_loop.append(next_edge)

                # Shuffle around for the next loop
                current_edge = next_edge

            # Check if the unordered collection of vertices is unique to the set of all unordered face vertices
            if set(vertex_loop) not in face_vertex_sets:
                # If it is, append the unordered set of vertices
                face_vertex_sets.append(set(vertex_loop))
                
                # Increment the faces per edge counter for each edge in the loop
                for e in edge_loop:
                    edge_face_counter[e.indices] += 1

                # Also append the ordered lists of vertices and edges
                face_vertex_loops.append([self[v] for v in vertex_loop])
                face_edge_loops.append(edge_loop)
                
            # If it is not, just advance to the next edge
            else:
                current_edge_index = next(edge_index_iter)
                
        print("Face generation complete")
                
        # When everything is done, return the collection of faces as vertex loops and edge loops
        return face_vertex_loops, face_edge_loops
             
    # Simple utility function to get a node given a set of coordinates, if possible
    def getNodeByCoord(self, coords, omit = list()):
        for n in self.nodes:
            if n.index in omit:
                continue
                
            if np.allclose(coords, n.coords, atol = 1e-2):
                return n
        return None
    
    # Draws the nodes and/or edges of the undirected graph as a plot
    def draw(self, percentile = 95, plot_dual_overlay = False, plot_nodes = True, plot_edges = True, plot_faces = False, plot_orbits = False, limit = True, plot_extra_pts = list(), fSize = (30, 30), kwargs = dict(), n_kwargs = dict(), e_kwargs = dict(), f_kwargs = dict(), pep_kwargs = dict()):
        fig, ax = self.plt.subplots(figsize = fSize)
            
        print("Starting draw")
        # Get the limits based on the upper and lower 10% quartiles
        coords = [n.coords for n in self.nodes]
        x_coords, y_coords = [c[0] for c in coords], [c[1] for c in coords]
        
        Q_x, Q_x = np.percentile(x_coords, 100 - percentile), np.percentile(x_coords, percentile)
        Q_y, Q_y = np.percentile(y_coords, 100 - percentile), np.percentile(y_coords, percentile)
    
        if plot_faces:
            faces_to_plot = self.faces if isinstance(plot_faces, bool) else [self.faces[i] for i in plot_faces]
            
            coll = [Polygon(f.plot_form, closed = True, **{**kwargs, **f_kwargs}) for f in faces_to_plot]
            
            coll = PatchCollection(coll, match_original = True) #cmap = plt.cm.Wistia) #
            #coll.set_array([f.area for f in faces_to_plot])
            
            ax.add_collection(coll)
        
        if plot_edges:
            edges_to_plot = self.edges if isinstance(plot_edges, bool) else [self.edges[i] for i in plot_edges]
            
            for e in edges_to_plot:
                if e.plot_form:
                    self.plt.plot(*e.plot_form, **{**kwargs, **e_kwargs})
                
        if plot_orbits:            
            for orbit in self.orbits:
                if len(list(orbit)) > 2:
                    o_kwargs = {'color' : np.random.random(3)}
                else:
                    o_kwargs = {'color' : 'gray', 
                                'marker' : 'x'}
                    
                for n in orbit:
                    self.plt.scatter(*self[n].coords, **{**kwargs, **n_kwargs, **o_kwargs})
        
        if plot_dual_overlay:
            n_kwargs = {'color' : 'red'}
            
            for n in self.dual.nodes:
                self.plt.scatter(*n.coords, **{**kwargs, **n_kwargs})
            
        if plot_nodes and not plot_orbits:
            nodes_to_plot = self.nodes if isinstance(plot_nodes, bool) else [self.nodes[i] for i in plot_nodes]
            
            for n in nodes_to_plot:
                self.plt.scatter(*n.coords, **{**kwargs, **n_kwargs})
                
        if len(plot_extra_pts) > 0:
            for p in plot_extra_pts:
                self.plt.scatter(*p, **{**kwargs, **pep_kwargs})                
        
        if limit:
            min_lim = min([Q_x, Q_y])
            max_lim = max([Q_x, Q_y])
            lim = (np.abs(min_lim) + np.abs(max_lim))/2.0
            
            self.plt.xlim([-lim, lim])
            self.plt.ylim([-lim, lim])
            
        self.plt.show()
        
    # Exports a csv file in the format compatible with Danny's code
    # Cell 1: x1 x2 x3 ... xn y1 y2 y3 ... yn
    # Cell 2: x1 x2 x3 ... xn y1 y2 y3 ... yn
    # ...
    # Cell M: x1 x2 x3 ... xn y1 y2 y3 ... yn
    def export_csv(self, filepath = "./", code = "aaa", width = 10, height = 10, depth = 10, angle = 0.0):
        
        filename = f"{filepath}{code}-{self.relative_density:.2f}-{width}x{depth}x{height}-{self.uniform_edge_thickness:.1f}-{angle:.1f}.c.csv"
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
            
            for face in self.faces:
                # Write x coordinates, then y coordinates
                writer.writerow(face.cad_form[0] + face.cad_form[1])
                
        print(f"CSV {filename} written!")
    
    def export_mesh(self, sim_dir = "./",
                    printer_tolerance = 0.0004,
                    cellSize = 0.01,
                    method = 'union',
                    method_kwargs = dict(),
                    fill_holes = False,
                    show_gmsh = True,
                    mesh = True, # Mostly a debug uption
                    bounds = None,
                    debug = False):
        
        # Check if the path exists, and create it if not
        if not os.path.exists(sim_dir):
            os.makedirs(sim_dir)
        
        gm.initialize() # Initialize gmsh
        gm.model.add("aperiodic")
        
        # Define the entity for the mesh
        lc = cellSize
        
        # 2D Case
        if self.dim == 2:
            # https://github.com/shapely/shapely/issues/990#issuecomment-698648542
            def simplify2(geom, dist, excludeExterior = False):
                if geom.geom_type == 'LinearRing':
                    return shp.LinearRing(geom.simplify(dist).coords[1:]).simplify(dist)
                elif geom.geom_type == 'Polygon':
                    if excludeExterior:
                        exterior = geom.exterior
                    else:
                        exterior = simplify2(geom.exterior, dist)
                    interiors = [simplify2(interior, dist) for interior in geom.interiors]
                    return shp.Polygon(exterior, interiors)
                elif (geom.geom_type.startswith('Multi')
                        or geom.geom_type == 'GeometryCollection'):
                    return type(geom)([simplify2(part, dist) for part in geom.geoms])
                else:
                    return geom.simplify(dist)            
            
            # Union method
            if method == 'union':
                # Define the struts in Shapely and take the union
                struts = [e.shape_form() for e in self.edges if e.shape_form()]

                # If there is a bounding Polygon for the graph, we add a thickened version of that too
                if self.bounds:
                    bounds_thick = self.bounds.exterior.buffer(2 * cellSize)#, join_style = 2)

                    struts = [bounds_thick] + struts

                tiling = unary_union(struts)

                # Simplify the mesh
                tiling.normalize()
                tiling = simplify2(tiling, printer_tolerance, excludeExterior = True)
                
            elif method == 'buffer':
                # Define the struts in Shapely and take the union
                struts = [e.shape_form(force_line = True) for e in self.edges if e.shape_form(force_line = True)]
                
                if self.bounds:
                    struts = [self.bounds.exterior] + struts
            
                struts = shp.MultiLineString(struts)
                
                # Make it extra chunky first
                method_kwargs['distance'] *= 2
                tiling = struts.buffer(**method_kwargs)
                
                # Then erode
                method_kwargs['distance'] *= -0.5
                tiling = tiling.buffer(**method_kwargs)
            
            else:
                raise Exception("Method not found")
            
            # Global counter variable for gmsh
            C = 1
            
            # Track the physical group IDs
            physicalGroups = dict()
            
            if isinstance(tiling, shp.MultiPolygon):
                shapes = tiling.geoms
            else:
                shapes = [tiling,]
                
            for s in range(len(shapes)):
                shape = shapes[s]
                
                boundaries = [shape.exterior,] + list(shape.interiors)
                
                curveloops = list()
                exterior_edges = list()
                hole_edges = list()
                
                for j in range(len(boundaries)):
                    bndry = boundaries[j]
                    
                    # Get the bounding boxes of the particular boundary we are working on
                    if bounds:
                        xmin, ymin = bounds[0]
                        xmax, ymax = bounds[1]
                    else:
                        xmin, ymin, xmax, ymax = bndry.bounds            
                    
                    # If j is zero we are dealing with the external boundary, and we simplify this to a box with tagged edges
                    if j == 0:                        
                        gm.model.geo.addPoint(xmin, ymin, 0.0, lc, C)
                        gm.model.geo.addPoint(xmax, ymin, 0.0, lc, C+1)
                        gm.model.geo.addPoint(xmax, ymax, 0.0, lc, C+2)
                        gm.model.geo.addPoint(xmin, ymax, 0.0, lc, C+3)
                        
                        gm.model.geo.addLine(C, C+1, C+4)
                        gm.model.geo.addLine(C+1, C+2, C+5)
                        gm.model.geo.addLine(C+2, C+3, C+6)
                        gm.model.geo.addLine(C+3, C, C+7)
                        
                        loop = list(range(C+4, C+8))
                        print(loop)
                        
                        gm.model.addPhysicalGroup(1, [C + 4,], C + 8, name = "Bottom")
                        gm.model.addPhysicalGroup(1, [C + 5,], C + 9, name = "Right")
                        gm.model.addPhysicalGroup(1, [C + 6,], C + 10, name = "Top")
                        gm.model.addPhysicalGroup(1, [C + 7,], C + 11, name = "Left")  
                        
                        physicalGroups["Bottom"] = C + 8
                        physicalGroups["Right"] = C + 9
                        physicalGroups["Top"] = C + 10
                        physicalGroups["Left"] = C + 11
                        
                        exterior_edges += loop
                            
                        C += 12
                        
                    # Otherwise we are dealing with a polygonal hole
                    else: 
                        # First check if the hole in question is smaller than the printer tolerance
                        deltaX, deltaY = xmax - xmin, ymax - ymin     
                    
                        # Skip it if so
                        if deltaX < printer_tolerance or deltaY < printer_tolerance:
                            pass
                        
                        else:
                            # Get the number of exterior points (and lines)
                            n_pts = len(bndry.coords) - 1

                            if debug:
                                print(f"Coordinates are {bndry.coords}")

                            # Record the indices of the start and end points
                            start_pt, end_pt = C, C + n_pts - 1

                            loop = list()

                            # Define the exterior points
                            for i in range(n_pts):
                                x, y = bndry.coords[i]

                                # Add the points to the model
                                gm.model.geo.addPoint(x, y, 0.0, lc, C)
                                if debug:
                                    print(f"Added point ({x}, {y}, 0.0) with index {C}")
                                C += 1

                            # Define the lines between the exterior points
                            for i in range(n_pts - 1):
                                gm.model.geo.addLine(start_pt + i, start_pt + i + 1, C)
                                if debug:
                                    print(f"Added line ({start_pt + i}, {start_pt + i + 1}) with index {C}")
                                loop.append(C)
                                C += 1

                            # Add the terminal line
                            gm.model.geo.addLine(end_pt, start_pt, C)
                            if debug:
                                print(f"Added line ({end_pt}, {start_pt}) with index {C}")
                            loop.append(C)
                            C += 1
                            
                            exterior_edges += loop

                    # Define the curve loop
                    gm.model.geo.addCurveLoop(loop, C)
                    if debug:
                        print(f"Added curve loop {loop} with index {C}")
                    curveloops.append(C)
                    C += 1
                            
                    if fill_holes and j != 0:
                        # Form the surfaces for the holes (if specified)
                        gm.model.geo.addPlaneSurface([C - 1,], C)
                        if debug:
                            print(f"Added planar surface {curveloops} with index {C}")
                        C += 1
                        
                        hole_surfaces.append(C-1)

                # Form the surface
                gm.model.geo.addPlaneSurface(curveloops, C)
                #physicalGroups["Body"] = C
                if debug:
                    print(f"Added planar surface {curveloops} with index {C}")
                C += 1
                
                # Synchronize
                gm.model.geo.synchronize()
                
                gm.model.addPhysicalGroup(2, [C - 1], C, name = "Body")
                physicalGroups["Body"] = C
                C += 1
                
                gm.model.addPhysicalGroup(1, exterior_edges + hole_edges, C, name = "Boundary")
                physicalGroups["Boundary"] = C
                C += 1
                        
                gm.model.addPhysicalGroup(1, hole_edges, C, name = "Hole Boundaries")
                physicalGroups["Hole Boundaries"] = C
                C += 1
                
                if fill_holes:
                    gm.model.addPhysicalGroup(2, hole_surfaces, C, name = "Hole Surfaces")
                    physicalGroups["Hole Surfaces"] = C
                    C += 1
                    
            
            # Synchronize
            gm.model.geo.synchronize()
                
            # Specify the algorithm
            gm.option.setNumber("Mesh.Algorithm", 1)
                
            # Generate
            if debug:
                print(f"Beginning mesh generation...")
            gm.model.mesh.generate(2)
            if debug:
                print(f"Mesh generation finished!")
            
            # Output
            gm.write(sim_dir + "/case.msh")
            
            if show_gmsh:
                # View
                gm.fltk.run()
            
            # Finalize
            gm.finalize()  
            
            return physicalGroups
            
    def simulate_displacement(self, physicalGroups, displacement, sim_dir):
        # Set the data dir
        #el.elmer.data_dir = sim_dir
        
        sim = el.elmer.Simulation()
        
        sim.header.update({"Include Path" : "\"\"",
                           "Results Directory" :  "\"\""})
        
        sim.settings = {
            "Max Output Level" : "5",
            "Coordinate System" : "Cartesian",
            "Coordinate Mapping(3)" : "1 2 3",
            "Simulation Type" : "Steady state",
            "Steady State Max Iterations" : "1",
            "Output Intervals(1)" : "1",
            "Solver Input File" : "case.sif",
            "Post File" : "case.vtu"
        }
        sim.constants = {  
            "Gravity(4)" : "0 -1 0 9.82",
            "Stefan Boltzmann" : "5.670374419e-08",
            "Permittivity of Vacuum" : "8.85418781e-12",
            "Permeability of Vacuum" : "1.25663706e-6",
            "Boltzmann Constant" : "1.380649e-23",
            "Unit Charge" : "1.6021766e-19"
        }
            
        # Define the PLA material
        PLA_XY = el.elmer.Material(sim, 'Ultimaker PLA (XY Orientation)')
        PLA_XY.data = {
            "Name" : "\"Ultimaker PLA (XY orientation)\"",
            "Density" : "1240.0",
            "Youngs modulus" : "3.25e9",  
            "Poisson ratio" : "0.35"              
        }
        
        # Set up the solver and equation
        solver_elasticity = el.elmer.Solver(sim, 'StressSolver')
        solver_elasticity.data.update({
            "Equation" : "Linear elasticity",
            "Calculate Stresses" : "True",
            "Calculate Loads" : "True",
            "Procedure" : "\"StressSolve\" \"StressSolver\"",
            "Exec Solver" : "Always",
            "Stabilize" : "True",
            "Optimize Bandwidth" : "True",
            "Steady State Convergence Tolerance" : "1.0e-5",
            "Nonlinear System Convergence Tolerance" : "1.0e-7",
            "Nonlinear System Max Iterations" : "20",
            "Nonlinear System Newton After Iterations" : "3",
            "Nonlinear System Newton After Tolerance" : "1.0e-3",
            "Nonlinear System Relaxation Factor" : "1",
            "Linear System Solver" : "Iterative",
            "Linear System Iterative Method" : "BiCGStab",
            "Linear System Max Iterations" : "500",
            "Linear System Convergence Tolerance" : "1.0e-10",
            "BiCGstabl polynomial degree" : "2",
            "Linear System Preconditioning" : "ILU0",
            "Linear System ILUT Tolerance" : "1.0e-3",
            "Linear System Abort Not Converged" : "False",
            "Linear System Residual Output" : "10",
            "Linear System Precondition Recompute" : "1"
        })
        
        solver_output = el.elmer.Solver(sim, 'SaveScalars')
        solver_output.data.update({
            "Exec Solver" : "After All",
            "Procedure" : "\"SaveData\" \"SaveScalars\"",
            "Filename" : "case.dat",
            "Save Variable 1" : "vonMises",
            "Save Variable 2" : "Stress",
            "Save Variable 3" : "Strain"
        })
        
        #solver_output = el.elmer.load_solver('ResultOutputSolver', sim)
        eqn = el.elmer.Equation(sim, "Elasticity", [solver_elasticity, solver_output])
        eqn.data.update({"Name" : "\"Elasticity\"", "Plane Stress" : "True", "Calculate Stresses" : "True"})
        
        # Define the body
        body = el.elmer.Body(sim, "Tiling", [physicalGroups["Body"]])
        body.name = "Tiling"
        body.material = PLA_XY
        body.equation = eqn
        
        # Define the boundary conditions
        bndry_fixed = el.elmer.Boundary(sim, "Fixed", [physicalGroups["Bottom"]])
        bndry_fixed.data.update({"Name" : "\"Static\"", "Displacement 1" : "0.0", "Displacement 2" : "0.0", "Displacement 3" : "0.0"})
        
        bndry_compress = el.elmer.Boundary(sim, "Compression", [physicalGroups["Top"]])
        bndry_compress.data.update({"Name" : "\"Compression\"", "Displacement 1" : f"{displacement[0]}", "Displacement 2" : f"{displacement[1]}", "Displacement 3" : f"{displacement[2]}"})
        
        # Export
        sim.write_startinfo(sim_dir)
        sim.write_sif(sim_dir)
        
        el.execute.run_elmer_grid(sim_dir, "case.msh")
        print("Grid Written")
        
        el.execute.run_elmer_solver(sim_dir) 
        print("Simulation Complete")
        
        err, warn, stats = el.post.scan_logfile(sim_dir)
        print("Errors:", err)
        print("Warnings:", warn)
        print("Statistics:", stats)       
    
    def set_uniform_edge_thickness(self, t = 0.0):
        for e in self.edges:
            e.thickness = t
            
        self.uniform_edge_thickness = t
            
            
    @property
    def centroid(self):
        return self._centroid0
    
    @property
    def total_face_area(self):
        return sum(f.area for f in self.faces)
    
    # Does not take into account overlaps at meeting points
    @property
    def total_edge_area_approx(self):
        return sum(e.area for e in self.edges)
    
    # Takes into account overlaps at meeting point
    # Can only handle cases with uniform edge thickness right now
    @property
    def total_edge_area_precise(self):
        # Start by calculating the area of the non-overlapping portions of the tiling
        area = sum(e.short_area for e in self.edges)
        
        # At each node there is now a region where several rectangles come close to intersecting, but do not
        # We take the convex hull of the vertices of these rectangles and sum the area of it
        for n in self.nodes:
            verts = list()
            
            for e in n.edges:
                # We want to get the vector of the edge FROM n TO its neighbour
                # That is, the vector pointed AWAY from n
                
                if e.indices[1] == n.index:
                    e = e.flipped
                
                # Parallel offset vector
                b = 2 * e.thickness * e.direction
                
                # Perpendicular offset vector
                c = e.thickness * e.normal
                
                verts.append(n.coords + b + c)  
                verts.append(n.coords + b - c)
            
            # Generate the convex hull
            CH = sp.spatial.ConvexHull(verts)
            
            # Add the volume
            area += CH.volume
            
        return area
    
    @property
    def edge_density(self):
        return self.total_edge_area_precise / self.total_face_area
    
    @property
    def relative_density(self):
        return self.edge_density
    
    # Returns the mean coordination number
    @property
    def mean_coordination(self):
        return sum([n.coordination for n in self.nodes])/len(self.nodes)
    
    @property
    def mean_length(self):
        return sum(e.length for e in self.edges)/len(self.edges)
    
    @property
    def mean_face_area(self):
        return sum(f.area for f in self.faces)/len(self.faces)   
    
    
    # Edge orientation distribution
    @property
    def EOD(self):
        # Generate the first time
        if self._EOD == None:
            # If we want the non-normalized data, we have to multiply by the edge lengths
            if self.normalized:
                orientations = [e.theta for e in self.edges]
            else:
                orientations = [e.length * e.direction for e in self.edges]

            self._EOD = DirectionalDistribution(orientations, polar = False)
            self._EOD.bin_data(num_bins = self.num_bins)
        
        return self._EOD
    
    def bounding_box_2D(self, bottom_left, upper_right):
        blx, bly = bottom_left[0], bottom_left[1]
        urx, ury = upper_right[0], upper_right[1]
        
        self.bounds = shp.Polygon([(blx, bly), (urx, bly), (urx, ury), (blx, ury)])

class Node:
    def __init__(self, index, coords, neighbours, graph):
        self.parent = graph
        self.index = index
        self.neighbours = neighbours
        self.raw_coords = np.array(coords)
        #self.coords = (self.parent.global_scale @ ((self.parent.global_rotation @ (np.array(coords) - self.parent.centroid)) + self.parent.centroid)) + self.parent.global_translation
        self.x, self.y = self.coords[0], self.coords[1]
        self.edges = list()
        self.faces = list()
        self.boundary = False

        self.orbit = frozenset()
        
        if len(self.coords) == 3:
            self.z = self.coords[2]
        
        self.coordination = len(self.neighbours)
        
    @property
    def coords(self):
        return ( self.parent.global_scale @ ((self.parent.global_rotation @ (self.raw_coords - self.parent.centroid)) + self.parent.centroid) ) + self.parent.global_translation
        #return self.raw_coords + self.parent.global_translation
        #return self.raw_coords - self.parent.centroid
    
    def __repr__(self):
        return f"Node {self.index}, Coordinates: {self.coords}, Neighbours: {self.neighbours}\n"
    
class Edge:
    def __init__(self, i, j, graph, virtual = False, thickness = 0.0):
        self.parent = graph
        self.dim = graph.dim
        self.indices = (i, j)
        self.nodes = (graph[i], graph[j])
        self.faces = list()
        self.boundary = False # Denotes if the edge is a border or internal edge
        
        self.forward_neighbours = [(i, n) for n in self.nodes[0].neighbours if n != j]
        self.backward_neighbours = [(n, j) for n in self.nodes[1].neighbours if n != i]
        
        #self.coords = [n.coords for n in self.nodes]
        self.virtual = virtual
        
        # Half the width of the edge if it were a solid strut
        # Use for generating meshes later
        self.thickness = thickness
        
        # Young's modulus for PLA in Pascals
        self.elasticity = 3.25e9 #4.107e9
        
        self.length = np.linalg.norm(self.coords[1] - self.coords[0])
        
        # Vector form
        self.vector = self.coords[1] - self.coords[0]
        self.direction = (self.coords[1] - self.coords[0]) / np.linalg.norm(self.coords[1] - self.coords[0])
        
        # A normalized vector perpendicular to the direction
        self.normal = self.direction[::-1]
        self.normal[1] = -self.normal[1]
        
        if self.dim == 2:
            x, y = self.coords[1][0] - self.coords[0][0], self.coords[1][1] - self.coords[0][1]
            
            # Polar angle in the range [-pi, pi]
            self.theta = np.arctan2(y, x) # Calculation for polar theta in 2D
            
            # Unsigned angle in the range [0, 2 pi]
            self.theta_u = self.theta if self.theta >= 0 else self.theta + 2 * np.pi
            
        elif self.dim == 3:
            pass
        
        else:
            raise Exception("Dimensions higher than 3 currently do not have an orientation implemented.")
           
    # Takes a nodal index (i.e. an endpoint) and returns the angle of the edge with respect to that node
    def angle_wrt_node(self, ni, unsigned = False):
        if ni not in self.indices:
            raise IndexError(ni)
            
        else:
            if not unsigned:
                if ni == self.indices[0]:
                    return self.theta
                
                else:
                    return self.theta + np.pi if self.theta < 0 else self.theta - np.pi
                
            else:
                if ni == self.indices[0]:
                    return self.theta_u
                
                else:
                    return self.theta_u - np.pi if self.theta_u >= np.pi else self.theta_u + np.pi

    # Returns the angle between the current edge and the given edge e2
    def angle_between(self, e2):
        phi = np.dot(self.vector, e2.vector) / (self.length * e2.length)
        
        if self.indices[0] == e2.indices[0] or self.indices[1] == e2.indices[1]:
            return np.arccos(phi)
        else:
            return np.arccos(-phi)

                    
    @property
    def coords(self):
        return [n.coords for n in self.nodes]
    
    @property
    def flipped(self):
        return self.parent[self.indices[::-1]]
        
    def __repr__(self):
        state = "Virtual" if self.virtual else "Real"
        
        return f"{state} Edge {self.indices}, Coordinates: {self.coords}, Length: {self.length:.3f}, Direction: {self.direction}, Theta: {self.theta:.3f},\n"
    
    @property
    def plot_form(self):
        if self.dim == 2:
            p1 = self.coords[0]
            p2 = self.coords[1]
                
            # Form the coordinates
            if self.thickness == 0.0:
                coords = (p1, p2)
                
                if self.parent.bounds:
                    snubEdge = self.parent.bounds.intersection(shp.LineString(coords))
                    
                    if snubEdge:
                        coords = snubEdge.coords
                        
                    else:
                        return None
            
            else:
                coords = (p1 + self.thickness * self.normal,
                          p1 - self.thickness * self.normal, 
                          p2 - self.thickness * self.normal,
                          p2 + self.thickness * self.normal,
                          p1 + self.thickness * self.normal)
                
                if self.parent.bounds:
                    snubEdge = self.parent.bounds.intersection(shp.Polygon(coords))
                    
                    if snubEdge:
                        if isinstance(snubEdge, shp.Polygon):
                            coords = snubEdge.exterior.coords
                            
                        # For degenerate cases
                        else:
                            return None                           
                        
                    else:
                        return None
                
            return ([c[0] for c in coords], [c[1] for c in coords])
                          
        
        elif self.dim == 3:
            pass
        
    @property
    def cad_form(self):
        if self.plot_form:
            return list(zip(*self.plot_form))
        else:
            return None
        
    def shape_form(self, force_line = False):
        if self.cad_form:
            if self.thickness == 0.0:
                return shp.LineString(self.cad_form)
            
            elif force_line:
                tmp = self.thickness
                self.thickness = 0.0
                
                out = shp.LineString(self.cad_form)
                
                self.thickness = tmp
                
                return out
                
            else:
                return shp.Polygon(self.cad_form)
                    
        else:
            return None
        
    @property
    def area(self):
        return self.length * (2 * self.thickness)
        
    @property
    def short_area(self):
        return (self.length - 4 * self.thickness) * (2 * self.thickness)
        
    def spring_constant(self, depth = 0.1):
        # Hooke's Law: sigma = E * epsilon,
        # ==> F / A = E * (delta_L / L)
        # ==> F = (A E / L) delta_L = k x
        # ==> Spring constant k = A E / L
        # A is cross-sectional area, E is the Young's modulus, L is the length
        
        # Depth = 0.1 corresponds to a 10cm = 0.1m depth sample cube
        
        return (depth * (2 * self.thickness)) * self.elasticity / self.length      
        
    # For compatability with CK's code
    def x(self):
        return self.plot_form[0]
    
    def y(self):
        return self.plot_form[1]
    
    def z(self):
        return self.plot_form[2]
        

class ReversedEdge(Edge):
    def __init__(self, i, j, graph):
        # Initialize the Edge class
        super().__init__(i, j, graph)
        
        self.indices = self.indices[::-1]
        self.nodes = self.nodes[::-1]
        
        self.thickness = graph[i, j].thickness
        
        self.forward_neighbours, self.backward_neighbours = self.backward_neighbours, self.forward_neighbours
        
        self.vector = -self.vector
        self.direction = -self.direction        
        
        # Flip the angle
        if self.dim == 2:
            self.theta = self.theta - np.pi if self.theta >= 0 else self.theta + np.pi   
            
            self.theta_u = self.theta_u - np.pi if self.theta_u >= np.pi else self.theta_u + np.pi
            
    @property
    def flipped(self):
        return self.parent[self.indices[::-1]]
            
    def __repr__(self):        
        return f"Reversed Edge {self.indices}, Coordinates: {self.coords}, Length: {self.length:.3f}, Direction: {self.direction}, Theta: {self.theta:.3f},\n"
    
class Face:    
    def __init__(self, vertex_loop, edge_loop, graph):
        self.parent = graph
        self.nodes = vertex_loop # Vertices in CCW order
        self.node_indices = [n.index for n in self.nodes]
        
        self.edges = edge_loop # Edges in CCW order
        self.edge_indices = [e.indices for e in self.edges]
        
        self.coords = [n.coords for n in self.nodes]
        
        # Cyclic Iterators
        self.CCW_vertex_cycler = itertools.cycle(self.nodes)
        self.CW_vertex_cycler = itertools.cycle(self.nodes[::-1])
        
        self.CCW_edge_cycler = itertools.cycle(self.edges)
        self.CW_edge_cycler = itertools.cycle(self.edges[::-1])
        
        for e in self.edges[:-1]:
            self.nodes.append(e.nodes[1])
            
        # Extra parameters for compatability with CK's code
        if graph.dim == 2:
            self.cx, self.cy = self.centroid[0], self.centroid[1]
            
            self.px = [c[0] for c in self.coords]
            self.py = [c[1] for c in self.coords]
            
        self.side_lengths = [e.length for e in self.edges]
        self.internal_angles = [e[0].angle_between(e[1]) for e in zip_offset(self.edges, self.edges, offsets = (0, 1), longest = True, fillvalue = self.edges[0])]
            
    @property
    def plot_form(self):
        return np.array(self.coords)    
    
    @property
    def cad_form(self):
        return list(zip(*self.coords))

    @property
    def shape_form(self):
        return shp.Polygon(self.coords)
            
    @property
    def area(self):
        """area = 0.5 * (self.nodes[-1].x * self.nodes[0].y - self.nodes[-1].y * self.nodes[0].x)
        
        for i in range(len(self.nodes) - 1):
            area += 0.5 * (self.nodes[i].x * self.nodes[i+1].y - self.nodes[i].y * self.nodes[i+1].x)
            
        return np.abs(area)"""
        return self.shape_form.area
    
    @property
    def centroid(self):
        return sum(self.coords)/len(self.coords)
    
    @property 
    def boundingBox(self):
        x, y = [v[0] for v in self.coords], [v[1] for v in self.coords]
        
        return ((min(x), min(y)),(max(x), max(y)))
    
    @property
    def star_volume(self):
        # TO DO
        return 0
    
    # Mean intercept length
    # NOT CORRECT AS OF 22/11/22. LOOK UP DEFINITION OF MIL AND CORRECT
    def MIL(self, samples = 360):
        theta = np.linspace(-np.pi, np.pi, num = samples)
        
        origin = self.centroid
        rays = (np.cos(theta), np.sin(theta))
        
        mean = 0.0
        
        for r in zip(*rays):            
            for e in self.edges:
                x0, y0 = origin[0], origin[1]
                rx, ry = r[0], r[1]
                vx, vy = e.nodes[0].coords[0], e.nodes[0].coords[1]
                ex, ey = e.direction[0], e.direction[1]
                
                if (rx * (ey - vy) - ry * (ex - vx)) != 0:
                    b = (ry * (vx - x0) - rx * (vy - y0)) / (rx * (ey - vy) - ry * (ex - vx))
                    
                    if 0 <= b <= 1:
                        mean += np.abs(((vx - x0) + b * (ex - vx))/rx) / np.sqrt(2)
                        
        mean /= samples
        return mean
    
    # Minimal area difference circumscribed ellipse
    # Need to investigate whether this is well-defined
    @property
    def MAD_circumscribed_ellipse(self):
        # TO DO
        return 0
    
    @property
    # Minimal area difference inscribed ellipse
    # Need to investigate whether this is well-defined
    def MAD_inscribed_ellipse(self):
        # TO DO
        return 0

    # Checks if a given face f2 is congruent to f1
    def is_congruent(self, f2):
        if len(self.nodes) != len(f2.nodes):
            return False

        else:
            # Not a valid check, but good enough for testing purposes
            return isclose(self.area, f2.area, rel_tol = 1e-04)
            
            #return sum(self.side_lengths) == sum(f2.side_lengths) and sum(self.internal_angles) == sum(f2.internal_angles)
        #return are_circularly_identical(self.side_lengths, f2.side_lengths) and are_circularly_identical(self.internal_angles, f2.internal_angles)       
    
class DirectionalDistribution:
    import matplotlib.pyplot as plt
    
    def __init__(self, data, polar = False):
        self.orientations = data  
        self.unique_orientations = set(self.orientations)
        self.unique_orientation_count = len(self.unique_orientations)
        
        if not polar:
            self.orientations += [theta - np.pi for theta in self.orientations if theta >= 0.0]
            self.orientations += [theta + np.pi for theta in self.orientations if theta < 0.0]        
        
        self._binned_data = list()
        
    def bin_data(self, num_bins = 360):
        # Get the buckets
        buckets = np.linspace(-np.pi, np.pi, num = num_bins, endpoint = False)
        
        # Get the bucket width
        bucket_width = buckets[1] - buckets[0]

        orientations = sorted(self.orientations)
        
        # Orientation distribution
        self._binned_data = [0 for i in range(num_bins)]
        i, j = 0, 0
        
        while i < num_bins - 1 and j < len(orientations):            
            if buckets[i] <= orientations[j] and orientations[j] < buckets[i + 1]:
                self._binned_data[i] += 1
                j += 1
            else:
                i += 1
        
        # Add the final values that have not been assigned anywhere else
        self._binned_data[-1] += len(orientations) - j
                
    @property
    def binned_data(self):
        if len(self._binned_data) == 0:
            self.bin_data()
            
        return self._binned_data
    
    def draw2d(self, histogram = True, lagrange_interpolate = False, FFT_interpolate = False, interp_max_deg = 10, interp_points = 1000, fSize = (16, 16), kwargs = dict()):
        fig = self.plt.figure(figsize = fSize)
        ax = self.plt.subplot(111, polar = True)
        #fig, ax = self.plt.subplots(figsize = fSize)
            
        if len(self._binned_data) == 0:
                self.bin_data()
        
        plot_data = self._binned_data            
        
        N = len(plot_data)
        theta = np.linspace(-np.pi, np.pi, num = N, endpoint = False)
        
        
        if histogram:
            radii = plot_data
            width = (2 * np.pi) / N
            
            bars = ax.bar(theta, np.sqrt(radii), width = width, bottom = 0)
            #circular_hist(ax, np.array(self._binned_data), bins = 32, gaps = False)
            
        if lagrange_interpolate:
            f_theta = plot_data
            
            # Get the Lagrange interpolation
            PL = sp.interpolate.lagrange(theta, f_theta)
            
            # Get the plot points r-axis
            phi = np.linspace(-np.pi, np.pi, num = interp_points, endpoint = False)            
            
        if FFT_interpolate:
            fft = np.fft.fft(plot_data)/N
            a, b = fft.real, fft.imag
            
            # Number of non-zero (binned) orientations
            M = min([len(set(plot_data)), interp_max_deg])
            
            # Trig Polynomial Interpolation derived from the FFT
            f = lambda x : 0.5*a[0] + sum([a[j] * np.cos(j*x) for j in range(M)]) + sum([b[j] * np.sin(j*x) for j in range(M)])
            
            # Plot the circle using the interpolation
            phi = np.linspace(-np.pi, np.pi, num = interp_points, endpoint = False)
            f_phi = f(phi)
            
            self.plt.plot(phi, f_phi)
            self.plt.title(f"Interpolation using a degree {M} trig polynomial derived from FFT")
            
        self.plt.show()

class LaguerreComplex:
    import laguerre as la
    
    def __init__(self, node_coordinates, symmetry = '', remove_outliers = True):
        self.laguerre = self.la.laguerre_tess(pts = node_coordinates, rad = 1.0)
        
        self.graph, self.centres, self.radii, self.LVerts, self.LEdgesIdx, self.LEdges, self.LRays = None, None, None, None, None, None, None   
        
        self.symmetry = symmetry
        
        self.regenerate()             
        
    def update_graph(self):
        self.graph = UndirectedGraph(self.LVerts, self.LEdgesIdx, symmetry = self.symmetry, dual = None)
        
    def regenerate(self):        
        # Centres, Radii, LVerts, LEdgesIdx, LEdges, LRays
        tmp = self.laguerre.generate()
        self.centres = tmp["Centres"]
        self.radii = tmp["Radii"]
        self.LVerts = tmp["LVerts"]
        self.LEdgesIdx = tmp["LEdgesIdx"]
        #self.LEdges = tmp["LEdges"] # Unused
        #self.LRays = tmp["LRays"] # Unused
        self.LRaysIdx = tmp["LRaysIdx"]
        
        self.update_graph()
        
    def box_data(self, debug = False):
        ndim = len(self.LVerts[0])
        dim_range = range(ndim)
        
        nnodes = len(self.LVerts)
        node_range = range(nnodes)
        
        # Get the minima and maxima for each coordinate
        minima = [min(v[i] for v in self.LVerts) for i in dim_range]
        maxima = [max(v[i] for v in self.LVerts) for i in dim_range]
        
        # Get the list of all vertices with coordination number <= 2
        valid_nodes = list()
        counter = [0 for i in node_range]
        
        for i in node_range:
            for e in self.LEdgesIdx:
                if i in e:
                    counter[i] += 1
                    
        #print(f"Counter is {counter}")
                    
        invalid_nodes = [i[0] for i in enumerate(counter) if i[1] <= 2]
        
        #print(f"Number of nodes found with less than three coordination: {len(invalid_nodes)}.")
        
        new_nodes_idxs = list()
        new_nodes = list()
        new_edges = list()
        degenerate_nodes = list()
        semidegenerate_nodes = list()
        
        for n0_idx in invalid_nodes:
            # Get the distance from n to each of the extremal boundarie
            n0 = np.array(self.LVerts[n0_idx])
            
            s = list()
        
            # If m lies on the data boundary then m = n0 + s d for some (non-negative) s
            # ==> s = (m[i] - n[i])/d[i] for each dimension
            # ==> s = m[i] - n[i] for each dim
            for j in dim_range:
                b = np.array((0.0,)*ndim)
                
                smin = np.abs(minima[j] - n0[j])
                smax = np.abs(maxima[j] - n0[j])
                
                if smin < smax:
                    b[j] += minima[j] - n0[j] #smin
                else:
                    b[j] += maxima[j] - n0[j] #smax
                    
                s.append(b)
                
            # Sort s by whichever results in the shortest distance
            s = min(s, key = lambda x : np.linalg.norm(x))
            #print(f"s is {s}")
            
            # If s is zero, then the node already lies on the data boundary
            if np.all(s == 0.0):
                degenerate_nodes.append(n0_idx)
                semidegenerate_nodes.append(n0_idx)
                if debug:
                    print(f"Border node {n0_idx} found.")
                #pass
            
            # If s is non-zero, we want to create a new node and a new edge
            else:
                n1 = n0 + s
                n1_idx = -1
                
                # New node already exists
                for idx, n in enumerate(self.LVerts + new_nodes):
                    if np.allclose(n, n1):
                        # Get the index of the existing node and break the loop
                        n1_idx = idx
                        degenerate_nodes.append(n1_idx)
                        #print(f"The node {n1} already exists.")
                        break
                    
                # New node does not already exist, so we create it
                if n1_idx == -1:
                    new_nodes.append(n1)
                    n1_idx = nnodes + len(new_nodes) - 1
                    degenerate_nodes.append(n1_idx)
                    #print(f"The node {n1} has been created.")
                    
                # Define the edge
                e = [n0_idx, n1_idx]
                
                if (e not in self.LEdgesIdx + new_edges) and (e[::-1] not in self.LEdgesIdx + new_edges):
                    new_edges.append(e)
                 
        self.LVerts += new_nodes
        
        # Create an additional set of nodes at the extremal corners of the data set
        for c_node in product(*zip(minima, maxima)):
            # Check the node doesn't already exist
            
            if np.any([np.allclose(c_node, n) for n in self.LVerts]):
                #print(f"Corner node {c_node} already exists at {self.LVerts.index(c_node)}")
                pass
            
            else:
                #print(f"Corner node {c_node} does not already exist.")
                self.LVerts.append(c_node)
        
        # Next we need to ensure there are no degenerate nodes with only a single edge connected to them
        #print(f"Number of degenerate nodes to be correct {len(degenerate_nodes)}.")
        
        for n0_idx in degenerate_nodes:
            # Get the initial node
            n0 = np.array(self.LVerts[n0_idx])
            nnodes = len(self.LVerts)
            
            dist_by_idx = lambda i : np.linalg.norm(n0 - self.LVerts[i])
            
            # We need to obtain all the nodes that are a cardinal translation away from n0
            # Split those depending direction (the non-zero dimension, and the sign of that dimension)
            # Sort everything by distance
            # Pick the two nearest cardinals in differing directions
            # Form an edge if possible
            # Loop if note
            
            # Get all the nodes that are a cardinal translation from n0
            cardinals = [i for i in range(nnodes) if sum(np.isclose(n0 - np.array(self.LVerts[i]), 0.0)) == ndim - 1]
            #print(f"The cardinals of {n0} are {[self.LVerts[i] for i in cardinals]}")
            
            # Cardinals_by_direction will be a list of the nearest neighbours in each cardinal direction
            cardinals_by_direction = list()
            
            #Sort by direction
            for i in range(ndim):
                pos, neg = list(), list()
                
                for j in cardinals:
                    # Displacement vector between the two
                    d = self.LVerts[j] - n0
                    
                    # Skip if zero on this axis
                    if d[i] == 0.0:
                        pass
                    
                    # Append to pos if positive
                    elif d[i] > 0.0:
                        pos.append(j)
                        
                    # Append to neg if negative
                    else:
                        neg.append(j)
                        
                # Select the closest vectors in pos and neg respectively  
                if len(pos) > 0:
                    pos =  min(pos, key = lambda k : self.LVerts[k][i])  
                    cardinals_by_direction.append(pos)  
                
                if len(neg) > 0:
                    neg =  max(neg, key = lambda k : self.LVerts[k][i]) 
                    cardinals_by_direction.append(neg) 
            
            # Now sort the cardinals by whichever are closest
            if len(cardinals_by_direction) > 0:
                cardinals_by_direction = sorted(cardinals_by_direction, key = lambda i : np.linalg.norm(n0 - self.LVerts[i]))
                
            # Stop when this is two
            new_edge_count = 0
                
            for n1_idx in cardinals_by_direction:
                if ([n0_idx, n1_idx] not in self.LEdgesIdx + new_edges) and ([n1_idx, n0_idx] not in self.LEdgesIdx + new_edges):
                    new_edges.append([n0_idx, n1_idx])
                    new_edge_count += 1
                    
                if n0_idx in semidegenerate_nodes and new_edge_count == 1:
                    break
                    
                if new_edge_count == 2:
                    break
                    
            #print(f"Ending loop for degenerate {n0_idx} with {new_edge_count} new edges created.")
        
        degenerates_remaining = 0
        
        #print(f"Minima: {minima}\n Maxima: {maxima}")
        
        for n_idx, n in enumerate(self.LVerts):
            count = 0
            
            for e in self.LEdgesIdx + new_edges:
                if n_idx in e:
                    count += 1
                    
            if count < 3:
                if debug:
                    print(f"Node {n_idx} at coordinates {n} has coordination {count} and may be degenerate")
        
        if debug:
            print(f"Number of new nodes created: {len(new_nodes)}, number of new edges created: {len(new_edges)}")
            
        self.LEdgesIdx += new_edges
        self.update_graph()
        
    def convert_rays_to_edges(self):
        n_existing_nodes = len(self.LVerts)
        n_rays = len(self.LRaysIdx)
        n_edges = len(self.LEdgesIdx)
        
        if n_rays == 0:
            print("Tessellation has no rays!")
            
            return
        
        dim = len(self.LRaysIdx[0][1])
        
        # These form the boundaries of the data
        lower_bound = [min(v[i] for v in self.LVerts) for i in range(dim)]
        upper_bound = [max(v[i] for v in self.LVerts) for i in range(dim)]
        
        new_nodes = list()
        new_edges = list()
        
        # We start by defining a collection of new nodes wherever the ray, after some finite distance, intercepts the data boundary and does not intercept any nodes before then
        
        for j in range(n_rays):            
            # r is a tuple of two elements
            # The first is the nodal index of the source of the ray
            # The second is the (normalized) directional vector of the ray
            r = self.LRaysIdx[j]
            
            idx0 = r[0] # Index of the source node of the ray
            n0 = np.array(self.LVerts[idx0]) # Coordinate vector of the node of the ray
            
            d = np.array(r[1]) # Directional vector of the ray        
        
            # We now want to trace each ray from its source along its directional vector until it hits either a data boundary or another node
            # The (positive) multipliers required to reach the boundary or node will be collected in this list
            s_choices = list()
            
            # First we check if the ray intercepts any existing nodes
            for m_idx, m in enumerate(self.LVerts):               
                #m = n0 + s * d for some s ==> m[i] = n0[i] + s d[i] for each coord ==> s = (m[i] - n0[i]) / d[i] must match for all i
                s = [(m[i] - n0[i]) / d[i] for i in range(dim) if d[i] != 0.0]
                
                # If all the values of s are equal, we have a possible candidate
                if np.all(s == s[0]):
                    
                    # If it is negative or zero, then we ignore the case. We only include positive, non-zero values of s
                    if s[0] <= 0.0:
                        pass

                    else:
                        # If s is positive and non-zero, we still need a secondary check to account for cases where d[i] == 0.0 for some coordinate
                        v1 = n0 + s[0] * d

                        # If all the coordinates match, we have a viable multiplier s
                        if np.all(m == v1):                    
                            # print(f"The ray {r} intersects node {m_idx} when the multiplier is {s[0]}")
                            s_choices.append(s[0])
                
            # Next we check when the ray intercepts any boundaries
            for b_dim, b in enumerate(zip(lower_bound, upper_bound)):
                # If the directional vector is zero in the current axis, then no s is going to extend the ray to a boundary of that axis
                if d[b_dim] == 0.0:
                    pass
                
                # Otherwise we get whichever of the upper and lower boundaries is positive
                else:
                    lb, ub = b[0], b[1]
                
                    s_lower = (lb - n0[b_dim]) / d[b_dim]
                    s_upper = (ub - n0[b_dim]) / d[b_dim]
                    
                    if s_lower > 0.0:
                        s_choices.append(np.abs(s_lower))
                        
                    if s_upper > 0.0:
                        s_choices.append(np.abs(s_upper))
            
            # If s_choices is non-zero then there are values of s such that the ray, when followed for a finite distance, intercepts either another node or a boundary
            if len(s_choices) > 0:
                s = min(s_choices)
                
                # Define a new node
                n1 = n0 + s * d
                
                # Append it to the list
                new_nodes.append(n1)
                
                # Get what its index will be, once added to the total list
                idx1 = len(self.LVerts) + len(new_nodes) - 1
                
                # Any new nodes are going to need at least two edges defined to avoid being degenerate
                # The first is, naturally, the truncated ray itself
                e_ray = [idx0, idx1]
                new_edges.append(e_ray)
                
                # For the second, we join the new node to its nearest neighbour that isn't n0
                NN = min( [pair for pair in enumerate(self.LVerts + new_nodes) if pair[0] not in (idx0, idx1)], key = lambda p : np.linalg.norm(n1 - p[1]) )
                
                e_nn = [idx1, NN[0]]
                new_edges.append(e_nn)
                
                #print(f"The nearest neighbour for the new node {n1} with index {idx1} is the existing node {NN[1]}" + 
                #      f" with index {NN[0]} and is a distance {np.linalg.norm(n1 - NN[1])} away.")
                    
        self.LVerts += new_nodes
        self.LEdgesIdx += new_edges
            
        self.update_graph()
        
    def find_and_remove_outliers(self, p = 95):
        self.remove_nodes(self.find_outliers(percentile = p))   
        
        self.update_graph()
    
    def find_outliers(self, percentile = 95):
        data = self.LVerts
        
        # Get the limits based on the upper and lower 10% quartiles
        ndata = len(data)
        ndim = len(data[0])

        split_coords = [[c[i] for c in data] for i in range(ndim)]

        percentiles = [(np.percentile(sc, 100-percentile), np.percentile(sc, percentile)) for sc in split_coords]

        outliers = list()

        for i in range(ndata):
            valid = True

            for j in range(ndim):
                if data[i][j] <= percentiles[j][0] or data[i][j] >= percentiles[j][1]:
                    valid = False

            if not valid:
                outliers.append(i)

        return outliers
    
    def remove_nodes(self, node_indices, debug = False):
        nnodes = len(node_indices)
        
        # Easier to do this when the nodes are sorted
        nodes_to_remove = sorted(node_indices, reverse = True)
        if debug:
            print(f"Nodes to remove {nodes_to_remove}")
        
        # Rays are the easiest to remove
        new_rays = list()
        
        for r in self.LRaysIdx:
            if r[0] not in nodes_to_remove:
                new_rays.append(r)
        
        # Loop controller
        degenerate = True
        
        curr_nodes = self.LVerts
        curr_edges = self.LEdgesIdx
        curr_rays = self.LRaysIdx
        
        while degenerate:
            new_edges = list()
            new_rays = list()
            
            # The hard part is restructing the edges
            # Start by filtering out the edges that are connected to an invalid node
            for e in curr_edges:
                invalid = False

                for idx in nodes_to_remove:
                    # Store the invalid edges in a separate list
                    if e[0] == idx or e[1] == idx:
                        invalid = True
                        break

                if not invalid:
                    new_edges.append(e)        

            # Now we need to decrement the indices of all the edges and rays corresponding to how many nodes before that index have been removed
            for e in new_edges:
                for idx in nodes_to_remove:
                    if e[0] > idx:
                        e[0] -= 1

                    if e[1] > idx:
                        e[1] -= 1
                    
            # Do the same for rays
            for r in curr_rays:
                invalid = True
                
                for idx in nodes_to_remove:
                    if r[0] == idx:
                        invalid = True
                        break
                        
                if not invalid:
                    new_rays.append(r)
            
            for r in new_rays:
                for idx in nodes_to_remove:
                    if r[0] > idx:
                        r[0] -= 1

            # Remove any self-connected edges
            new_edges = [e for e in new_edges if e[0] != e[1]]

            # Finally remove the invalid nodes
            new_nodes = [curr_nodes[i] for i in range(len(curr_nodes)) if i not in nodes_to_remove]

            # Next we perform a pass to check if this process has created any degenerate nodes (nodes with a single edge)
            used_nodes = { i : 0  for i in range(len(new_nodes))}

            for e in new_edges:
                used_nodes[e[0]] += 1
                used_nodes[e[1]] += 1

            nodes_to_remove = sorted([i for i in used_nodes.keys() if used_nodes[i] <= 1], reverse = True)
            
            if len(nodes_to_remove) == 0:
                degenerate = False
            else:
                if debug:
                    print(f"Invalid Nodes: {nodes_to_remove}")
                
            # Update for the next loop
            curr_edges = new_edges
            curr_nodes = new_nodes
            curr_rays = new_rays
        
        if debug:
            print(f"Number of nodes removed: {len(self.LVerts) - len(new_nodes)}")
            print(f"Number of edges removed: {len(self.LEdgesIdx) - len(new_edges)}")
            print(f"Number of rays removed: {len(self.LRaysIdx) - len(new_rays)}")
        
        self.LVerts = curr_nodes
        self.LEdgesIdx = curr_edges  
        self.LRaysIdx = curr_rays   
        
    # Scales and offsets the data to fit inside a box specified by the bottom-left and top-right coordinates
    def scale_to_fit(self, bottom_left, top_right, keep_aspect = True, ):
        # Dimensional indices
        ndim = len(self.LVerts[0])
        dim_range = range(ndim)
        
        # Vertex count indices
        nnodes = len(self.LVerts)
        node_range = range(nnodes)
        
        # Get the minima and maxima for each coordinate in each dimension
        minima = [min(v[i] for v in self.LVerts) for i in dim_range]
        maxima = [max(v[i] for v in self.LVerts) for i in dim_range]
        
        # The offset is calculated so that the minima lie along the axes
        offset = np.array([-m for m in minima])
        
        # The scale is calculated so that the data fits inside a box
        scale = np.array([(top_right[i] - bottom_left[i]) / (maxima[i] - minima[i]) for i in dim_range])
        
        # Although if we want to preserve the aspect ratio, we take the smallest of the scale factors and use that for all dimensions
        if keep_aspect:
            scale = np.array((min(scale),)*ndim)
        
        updated_nodes = list()
        
        for n in self.LVerts:    
            n = list(n)
            
            for i in dim_range:
                n[i] += offset[i]
                n[i] *= scale[i]
                n[i] += bottom_left[i]
                
            updated_nodes.append(n)
            
        self.LVerts = updated_nodes                
        self.update_graph()
        
        #print(f" O : {offset}")
        #print(f" SR : {scale}")
        
    # Scales and offsets the data so that the box specified by the bottom-left and top-right coordinates fits completely inside it
    def scale_to_contain(self, bottom_left, top_right, keep_aspect = True):
        # Dimensional indices
        ndim = len(self.LVerts[0])
        dim_range = range(ndim)
        
        # Vertex count indices
        nnodes = len(self.LVerts)
        node_range = range(nnodes)
        
        #print(lir.rectangle([n.coords for n in self.graph.boundary_nodes]))
        
        """# Get the minima and maxima for each coordinate in each dimension
        minima = [min(n.coords[i] for n in self.graph.boundary_nodes) for i in dim_range]
        maxima = [max(n.coords[i] for n in self.graph.boundary_nodes) for i in dim_range]
        
        # The offset is calculated so that the minima lie along the axes
        offset = np.array([-m for m in minima])
        
        # The scale is calculated so that the data fits inside a box
        scale = np.array([(top_right[i] - bottom_left[i]) / (maxima[i] - minima[i]) for i in dim_range])
        
        # Although if we want to preserve the aspect ratio, we take the smallest of the scale factors and use that for all dimensions
        if keep_aspect:
            scale = np.array((min(scale),)*ndim)
        
        updated_nodes = list()
        
        for n in self.LVerts:    
            n = list(n)
            
            for i in dim_range:
                n[i] += offset[i]
                n[i] *= scale[i]
                
            updated_nodes.append(n)
            
        self.LVerts = updated_nodes                
        self.update_graph()"""
        
    
# Takes input data consisting of a list of n-dimensional coordinate, scales and translates them to fit a n-dimensional cube of given size with the origin at the bottom-left
def data_scaler(data, edge_length = 0.1, percentile = 95):
        # Get the limits based on the upper and lower 10% quartiles
        data_dim = len(data[0])
        
        split_coords = [[c[i] for c in data] for i in range(data_dim)]
        
        percentiles = [(np.percentile(sc, 100-percentile), np.percentile(sc, percentile)) for sc in split_coords]
        
        widths = [p[1] - p[0] for p in percentiles]
        max_width = max(widths)
        
        scale = edge_length / max_width
        
        # Translate the data so percentile percentage of points lie in a square with the origin at the lower-left
        T = np.array([-p[0] for p in percentiles])
        
        out_data = [scale * (d + T) for d in data]
        
        return out_data    
                      
from collections import deque
 
def are_circularly_identical(list1, list2):
    d = deque(list1)
    for _ in range(len(list1)):
        if all(a == b for a, b in zip(d, list2)):
            return True
        d.rotate(1)
    return False

def identify_vertex_stars(points):
    vor = sp.spatial.Voronoi(pts)

    for region in vor.regions:
        # Boundary region
        if -1 in region or len(region) == 0:
            continue

        else:
            return False
            