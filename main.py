import shutil
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy as sp
import subprocess
import os
import sys

###
### See README for more details        

### Direction to results folder (will require local copy of results)
cwd = os.getcwd()+'/'
## Manually entered, to automate can take last 3 lines from crys.crds or testout.rst
dim = np.asarray([57.3612480, 57.3612480, 57.3612480])

## select given run
results_pick = 0

def produce_r_cutoff(results_pick):


    ## Coordinates included atom by atom
    with open('results/{:}/finalpos.out'.format(results_pick), 'r') as f:
        n_crds = np.genfromtxt(f, max_rows=667)
    with open('results/{:}/finalpos.out'.format(results_pick), 'r') as f:
        o_crds = np.genfromtxt(f, max_rows=667, skip_header=667)
    with open('results/{:}/finalpos.out'.format(results_pick), 'r') as f:
        p_crds = np.genfromtxt(f, max_rows=667, skip_header=667+667)

    ## VMD file written out to visualise results
    with open('results/{:}/vmd_file.xyz'.format(results_pick), 'w') as f:
        f.write('{:}\n\n'.format(int(n_crds.shape[0]+o_crds.shape[0]+p_crds.shape[0])))
        for i in range(n_crds.shape[0]):
            f.write('N{:<5}{:<20}{:<20}{:<20}\n'.format(i, n_crds[i,0], n_crds[i,1], n_crds[i,2]))
        for i in range(o_crds.shape[0]):
            f.write('O{:<5}{:<20}{:<20}{:<20}\n'.format(i, o_crds[i, 0], o_crds[i, 1], o_crds[i, 2]))
        for i in range(p_crds.shape[0]):
            f.write('P{:<5}{:<20}{:<20}{:<20}\n'.format(i, p_crds[i, 0], p_crds[i, 1], p_crds[i, 2]))


    def pbc_r(i,j):
        atom_1_crds = p_crds[i,:]
        atom_2_crds = p_crds[j,:]
        v = np.subtract(atom_2_crds, atom_1_crds)
        for axis in range(3):
            if v[axis] > dim[axis]/2:
                v[axis] -= dim[axis]
            elif v[axis] < -dim[axis]/2:
                v[axis] += dim[axis]
        r = np.linalg.norm(v)
        return r

    ## Reduce data set to list of P-P distances (can change) under 12 a.u.
    r_list = []
    for p_atom_1 in range(p_crds.shape[0]-1):
        for p_atom_2 in range(p_atom_1+1, p_crds.shape[0]):
              r = pbc_r(p_atom_1, p_atom_2)
              if r<12:
                  r_list.append(r)

    ## Visualise and estimate nearest neighbour distances
    fig, ax = plt.subplots(figsize=(8, 4))

    n_bins = 20
    counts, bins = np.histogram(r_list, bins=n_bins)
    bin_gap = (bins[1]-bins[0])/2
    ax.hist(r_list, bins=n_bins)

    array_r = np.asarray(r_list)
    # find nearest neighbour and second nearest neighbour peak
    peaks, _ = sp.signal.find_peaks(counts, height=50)
    # visualise these
    plt.scatter(np.add(bins[peaks], bin_gap), peaks, color='orange')
    plt.plot([bins[0],bins[0]], [0,100], color='orange')
    list_counts = counts.tolist()
    # find local minima between these
    min_val = bins[list_counts.index(min(counts[peaks[0]:peaks[1]]))]
    plt.plot([min_val+bin_gap, min_val+bin_gap], [0,100], color='yellow')
    r_cutoff = min_val + bin_gap
    plt.show()
    return p_crds, r_cutoff
##################################################################################
def pbc_r(i,j):
    atom_1_crds = p_crds[i,:]
    atom_2_crds = p_crds[j,:]
    v = np.subtract(atom_2_crds, atom_1_crds)
    for axis in range(3):
        if v[axis] > dim[axis]/2:
            v[axis] -= dim[axis]
        elif v[axis] < -dim[axis]/2:
            v[axis] += dim[axis]
    r = np.linalg.norm(v)
    return r


p_crds, r_cutoff = produce_r_cutoff(results_pick)

def produce_Graph(r_cutoff):

    ## Graph stats

    G = nx.Graph()

    ## Add nodes to graph
    for p_atom in range(p_crds.shape[0]):
        G.add_node(p_atom, pos=(p_crds[p_atom,0], p_crds[p_atom,1], p_crds[p_atom,2]))


    print(r_cutoff)


    ## Add edges to graph
    for p_atom_1 in range(p_crds.shape[0]-1):
        for p_atom_2 in range(p_atom_1+1, p_crds.shape[0]):
            r = pbc_r(p_atom_1, p_atom_2)
            if r<r_cutoff:
                G.add_edge(p_atom_1, p_atom_2)

    def Rank(G):
        ## call to visualise P coordination numbers
        degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
        dmax = max(degree_sequence)

        fig = plt.figure("Degree of a random graph", figsize=(8, 8))
        # Create a gridspec for adding subplots of different sizes
        axgrid = fig.add_gridspec(5, 4)

        ax0 = fig.add_subplot(axgrid[0:3, :])
        Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
        pos = nx.spring_layout(Gcc, seed=10396953)
        nx.draw_networkx_nodes(Gcc, pos, ax=ax0, node_size=20)
        nx.draw_networkx_edges(Gcc, pos, ax=ax0, alpha=0.4)
        ax0.set_title("Connected components of G")
        ax0.set_axis_off()

        ax1 = fig.add_subplot(axgrid[3:, :2])
        ax1.plot(degree_sequence, "b-", marker="o")
        ax1.set_title("Degree Rank Plot")
        ax1.set_ylabel("Degree")
        ax1.set_xlabel("Rank")

        ax2 = fig.add_subplot(axgrid[3:, 2:])
        ax2.bar(*np.unique(degree_sequence, return_counts=True))
        ax2.set_title("Degree histogram")
        ax2.set_xlabel("Degree")
        ax2.set_ylabel("# of Nodes")

        fig.tight_layout()
        plt.show()
        return
    #Rank(G)

    from mpl_toolkits.mplot3d import Axes3D

    def plot_3d(G, list):

        # The graph to visualize connections in 3D

        # 3d spring layout
        pos = nx.spring_layout(G, dim=3, seed=779)
        pos = nx.get_node_attributes(G, 'pos')
        # Extract node and edge positions from the layout
        node_xyz = np.array([pos[v] for v in sorted(G)])
        edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])
        node_xyz_c = np.array([pos[v] for v in list])
        edge_xyz_c = np.array([(pos[list[i]], pos[list[i+1]]) for i in range(len(list)-1)])
        # Create the 3D figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Plot the nodes - alpha is scaled by "depth" automatically
        ax.scatter(*node_xyz.T, s=1, ec="w")
        ax.scatter(*node_xyz_c.T, s=100, ec="r")

        # Plot the edges
        for vizedge in edge_xyz:
            ax.plot(*vizedge.T, color="tab:gray")
        for vizegde in edge_xyz_c:
            ax.plot(*vizedge.T, color="tab:red")



        def _format_axes(ax):
            """Visualization options for the 3D axes."""
            # Turn gridlines off
            ax.grid(False)
            # Suppress tick labels
            for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
                dim.set_ticks([])
            # Set axes labels
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")

        _format_axes(ax)
        fig.tight_layout()
        plt.show()

    ## Calculate ring distribuition
    rings = {}



    print('Mean C.N. ', np.mean(sorted((d for n, d in G.degree()), reverse=True)))

    return G, np.mean(sorted((d for n, d in G.degree()), reverse=True))

G, mean_cn = produce_Graph(r_cutoff)

#sys.exit()

results_list = []
mean_list = []
for i in range(9):
    if i%2 ==0:
        results_pick = '{:}'.format(int(i*2.5))
    else:
        results_pick = '{:.1f}'.format(i*2.5)
    p_crds, r_cutoff = produce_r_cutoff(results_pick)
    G, mean_cn = produce_Graph(r_cutoff)

    results_list.append(i*2.5)
    mean_list.append(mean_cn)
plt.scatter(results_list, mean_list)
plt.show()

#sys.exit()
print('Val : [0-20]')
val = input()
p_crds, r_cutoff = produce_r_cutoff(val)
G, mean_cn = produce_Graph(r_cutoff)


import time
start = time.time()
A = nx.minimum_cycle_basis(G)
stop = time.time()
print(stop-start, ' seconds')
for c in A:
    print(len(c))
    pn.append(len(c))

with open('Ring_Distribution_{:}.dat'.format(0), 'w') as f:
    for i in range(14):
        f.write('{:<10}{:<10}\n'.format(i, pn.count(i)))



plt.plot([i for i in range(14)], [pn.count(i) for i in range(14)])
plt.show()

sys.exit()


neighbours = G.neighbors(0)
neighbours = [i for i in neighbours]
print(neighbours)

overall_string = 'index '

neighbour_ring = {}
neighbour_ring_lens = []
for i in range(len(neighbours)-1):
    for j in range(i+1, len(neighbours)):
        print(i,j)
        n1 = neighbours[i]
        n2 = neighbours[j]
        counter = 0
        for path in nx.shortest_simple_paths(G, source=n1, target=n2):

            rings['{:}'.format(counter)] = path
            counter += 1
            if counter > 2:
                break
        if 0 not in rings['1']:
            path = rings['1']
        else:
            path = rings['0']
        print(path)
        neighbour_ring['{:}{:}'.format(i,j)] = path
        neighbour_ring_lens.append(len(path))
        string = 'index '
        for val in path:
            string += '{:} or index '.format(int(val) + 1334)
            overall_string += '{:} or index '.format(int(val) + 1334)
        print(string)
    print('Overall string : ')
    print(overall_string)
print(neighbour_ring_lens)
import sys
sys.exit()
for neighbour in G.neighbors(0):
    counter = 0
    for path in nx.shortest_simple_paths(G, source=0, target=neighbour):

        rings['{:}'.format(counter)] = path
        counter +=1
        if counter >2:
            break

    path = rings['1']
    print(path)
    string = 'index '
    for i in path:
        string += '{:} or index '.format(int(i)+1334)
    print(string)
    plot_3d(G, path)
    print('\n\n#####################################\n\n')



## plot the cumulative histogram
#n, bins, patches = ax.hist(r_list, n_bins, density=True, histtype='step',
#                           cumulative=True, label='Empirical')
#
## Overlay a reversed cumulative histogram.
#ax.hist(r_list, bins=bins, density=True, histtype='step', cumulative=-1,
#        label='Reversed emp.')
#
## tidy up the figure
#ax.grid(True)
#ax.legend(loc='right')
#ax.set_title('Cumulative step histograms')
#ax.set_xlabel('Annual rainfall (mm)')
#ax.set_ylabel('Likelihood of occurrence')
#
#plt.show()




