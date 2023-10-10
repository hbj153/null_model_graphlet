import pandas as pd
import numpy as np
from tqdm import trange
from collections import defaultdict
from itertools import combinations
import random
from scipy.stats import bernoulli
from scipy import stats
import sys
import calibration
import matplotlib.pyplot as plt
import os
import networkx as nx


class Formatter:
    @staticmethod
    def edges_to_edgedict(edges_pos, edges_neg, exclude_self_loops: bool = True):
        """edges_to_edgedict(edges_pos,edges_neg): convert edges to edge dictionary
        Note if a edge has both positive and negative signs, we will remove it

        Parameters
        ----------
        edges_pos : list
            list of positive edges, example: [(1,2),(2,3)]
        edges_neg : list
            list of negative edges, example: [(1,2),(2,3)]
        exclude_self_loops : bool, optional
            whether to exclude self loops, by default True

        Returns
        -------
        edge_dict : dict
            dictionary of edges, example: {(1,2):1,(2,3):-1}
        """
        # conflict edges: an edge has both positive and negative signs
        edges_pos = [tuple(sorted(edge)) for edge in edges_pos]
        edges_neg = [tuple(sorted(edge)) for edge in edges_neg]
        conflict_edges = set(edges_pos).intersection(set(edges_neg))
        edges_dict = {}

        for edge in edges_pos:
            # exclude self loops
            if exclude_self_loops and edge[0] == edge[1]:
                continue
            if edge not in conflict_edges:
                edges_dict[tuple(sorted(edge))] = 1

        for edge in edges_neg:
            # exclude self loops
            if exclude_self_loops and edge[0] == edge[1]:
                continue
            if edge not in conflict_edges:
                edges_dict[tuple(sorted(edge))] = -1

        return edges_dict

    @staticmethod
    def df_to_edgedict(df, col1: str, col2: str, sign_col: str, directed: bool = False, exclude_self_loops: bool = True, nodeMapping: bool = False):
        """df_to_edgedict converts a dataframe to an edge dictionary. The incompatible edges are removed.
            The incompatible edges are defined as follows:
            1. for directed graph, remove edges with opposite signs
            2. for undirected graph, remove edges when (a,b) has opposite signs with (b,a), or (a,b) has both signs.

        Parameters
        ----------
        df : pd.Dataframe
            The network data in dataframe format. Columns: col1, col2, sign_col
        col1 : str
            The name of column represent node1.
        col2 : str
            The name of column represent node2.
        sign_col : str
            The name of column represent signs.
        directed : bool, optional
            if the graph is directed, by default False
        exclude_self_loops : bool, optional
            if exclude self loops, by default True
        nodeMapping : bool, optional
            if map node to node id, by default False

        Returns
        -------
        edges_dict : dict
            dictionary of edges, example: {(1,2):1,(2,3):-1}
        name2id : dict
            dictionary of node name to node id. Only return when nodeMapping=True
        id2name : dict
            dictionary of node id to node name. Only return when nodeMapping=True
        """
        df1 = df[[col1, col2, sign_col]]
        edges_dict_all = defaultdict(set)
        for row in df1.values:
            edges_dict_all[(row[0], row[1])].add(row[2])
        edges_dict = {}
        # 1. directed graph, only keep edges with one sign
        if directed:
            for e in edges_dict_all:
                if len(edges_dict_all[e]) == 1:
                    edges_dict[e] = edges_dict_all[e].pop()
        # 2. undirected graph, keep edges with the sign same for (a,b) and (b,a)
        else:
            for e in edges_dict_all:
                all_signs = edges_dict_all[e] | edges_dict_all.get(
                    e[::-1], set())
                if len(all_signs) == 1:
                    edges_dict[tuple(sorted(e))] = all_signs.pop()
        # remove self loops
        if exclude_self_loops:
            edges_dict = {e: edges_dict[e] for e in edges_dict if e[0] != e[1]}
        # node mapping
        node2id = {}
        id2node = {}
        if nodeMapping:
            nodes = set(sorted(pd.concat([df[col1], df[col2]])))
            for i, node in enumerate(nodes):
                node2id[node] = i
                id2node[i] = node
            edges_dict = {(node2id[e[0]], node2id[e[1]])
                           : edges_dict[e] for e in edges_dict}

        return edges_dict, node2id, id2node


class Motif:

    @staticmethod
    def find_triangles(edges_dict: dict, exclude_self_loops: bool = True):
        edges = list(edges_dict.keys())
        adj = defaultdict(set)
        for edge in edges:
            # skip self loops
            if edge[0] == edge[1] and exclude_self_loops:
                continue
            adj[edge[0]].add(edge[1])
            adj[edge[1]].add(edge[0])

        nodes_queue = list(adj.keys())
        triangles = []
        while nodes_queue:
            node = nodes_queue.pop()
            neighbors = adj[node]
            while adj[node]:
                neighbor = adj[node].pop()
                common_neighbors = adj[node] & adj[neighbor]
                # if node and neighbor have common neighbors, they can form a triangle
                if common_neighbors:
                    for common_neighbor in common_neighbors:
                        triangles.append((node, neighbor, common_neighbor))
                        # remove node from adj
                        if node in adj[neighbor]:
                            adj[neighbor].remove(node)
                        if node in adj[common_neighbor]:
                            adj[common_neighbor].remove(node)
                        if neighbor in adj[node]:
                            adj[node].remove(neighbor)
        return triangles

    @staticmethod
    def classify_triangles(triangles: list, edges_dict: dict):
        """classify_triangles This is a more efficient way to classify the number of triangles 
        with different number of positive edges compared to the exclude_automorphisms=True method in dotmotif.

        Parameters
        ----------
        triangles : list
            list of triangles, example: [(1,2,3),(2,3,4)]
        edges_dict : dict
            dictionary of edges, example: {(1,2):1,(2,3):-1}

        Returns
        -------
        res : dict
            dict of number of triangles with different number of positive edges
        """
        # a new edge dictionary with both directions (1,2) and (2,1) are included
        egdes_dict_double = {}
        for edge in edges_dict:
            egdes_dict_double[edge] = edges_dict[edge]
            egdes_dict_double[edge[::-1]] = edges_dict[edge]

        triangles_unique = {}
        # save the number of triangles of different positive edges
        res = {0: 0, 1: 0, 2: 0, 3: 0}
        for triangle in triangles:
            nodes = tuple(sorted(triangle))
            # if this is a new triangle, count positive edges
            if nodes not in triangles_unique:
                count_pos = 0
                for pair in combinations(nodes, 2):
                    if egdes_dict_double[pair] > 0:
                        count_pos += 1
                triangles_unique[nodes] = count_pos
                res[count_pos] += 1
        return res

    @staticmethod
    def find_squares(edges_dict: dict, exclude_self_loops: bool = True):
        """find_squares(edges_dict, exclude_self_loops=True): find all squares in the graph
        Parameters
        ----------
        edges_dict : dict
            dictionary of edges, example: {(1,2):1,(2,3):-1}
        exclude_self_loops : bool
            whether exclude self loops, default True

        Returns
        -------
        squares : list
            list of squares, example: [(1,2,3,4),(2,3,4,5)]
        """
        edges = list(edges_dict.keys())
        adjlist = defaultdict(set)
        for edge in edges:
            # skip self loops
            if edge[0] == edge[1] and exclude_self_loops:
                continue
            adjlist[edge[0]].add(edge[1])
            adjlist[edge[1]].add(edge[0])
        nodes_queue = list(adjlist.keys())
        squares = []
        while nodes_queue:
            node = nodes_queue.pop()
            neighbors = adjlist[node]
            for neighbor1, neighbor2 in combinations(neighbors, 2):
                # delete the node from the adjlist of its neighbors
                # if the node exists in the adjlist of its neighbors, remove it
                if node in adjlist[neighbor1]:
                    adjlist[neighbor1].remove(node)
                if node in adjlist[neighbor2]:
                    adjlist[neighbor2].remove(node)
                # check if the two neighbors has a common neighbor
                common_neighbors = adjlist[neighbor1] & adjlist[neighbor2]
                if len(common_neighbors) > 0:
                    for common_neighbor in common_neighbors:
                        # squares.append({'A':node,'B':neighbor1,'C':common_neighbor,'D':neighbor2})
                        squares.append(
                            (node, neighbor1, common_neighbor, neighbor2))
        return squares

    @staticmethod
    def classify_squares(squares: list, edges_dict: dict, graphletMode: bool = False):
        """classify_squares(squares, edges_dict): classify squares into 6 types.
        Parameters
        ----------
        squares : list
            list of squares, example: [(1,2,3,4),(2,3,4,5)]
        edges_dict : dict
            dictionary of edges, example: {(1,2):1,(2,3):-1}

        Returns
        -------
        res: dict
            dictionary of squares type count, example {0: 0, 1: 0, 21: 0, 22: 0, 3: 0, 4: 0}.
        """
        # a new edge dictionary with both directions (1,2) and (2,1) are included
        egdes_dict_double = {}
        for edge in edges_dict:
            egdes_dict_double[edge] = edges_dict[edge]
            egdes_dict_double[edge[::-1]] = edges_dict[edge]

        # save the results
        res = {0: 0, 1: 0, 21: 0, 22: 0, 3: 0, 4: 0}
        for square in squares:
            # --graphletMode: if the square has diagonal edges, skip it
            if graphletMode:
                if (square[0], square[2]) in egdes_dict_double or (square[1], square[3]) in egdes_dict_double:
                    continue
            # This saves time than iterating over the edges
            signs = [egdes_dict_double[(square[0], square[1])], egdes_dict_double[(square[1], square[2])], egdes_dict_double[(
                square[2], square[3])], egdes_dict_double[(square[3], square[0])]]
            pos = 0
            same_sign = 0  # denotes the number of adjacent edges with the same sign
            for i, sign in enumerate(signs):
                if sign > 0:
                    pos += 1
                # once the same_sign>0, it will not be +-+-
                if same_sign == 0 and signs[i] == signs[i-1]:
                    same_sign += 1
            # classify the square
            if pos == 2:
                if same_sign > 0:
                    res[21] += 1  # --++
                else:
                    res[22] += 1  # +-+-
            else:
                res[pos] += 1  # 63s
        return res

    @staticmethod
    def organize_squareZ(square: tuple, link: tuple):
        """
        organize_squareZ(square:tuple, link:tuple)
        Organize the squareZ motif into a list of nodes and edges.
        -------
        Parameters:
        square: tuple
            a square, for example (1,2,3,4)
        link: tuple
            a link, for example (1,3) or (2,4)
        -------
        Returns:
        squareZ: tuple
            a squareZ in the form of (1,2,3,4,1,3)

        Example, get_squareZ((2,8,1,4),(2,1)), there are two representations of the squareZ: (2,8,1,4,2,1) and (1,4,2,8,1,2). 
        The function will return the one with the smaller first element, i.e. (1, 4, 2, 8, 1, 2).
        """

        a, b, c, d = square
        # invalid squareZ
        if link != (a, c) and link != (a, c)[::-1] and link != (b, d) and link != (b, d)[::-1]:
            return None
        m, n = link
        # start with the smaller one
        head = min([m, n])
        headidx = square.index(head)
        sorted_square = square[headidx:] + square[:headidx]
        squareZ = (sorted_square[0], sorted_square[1], sorted_square[2],
                   sorted_square[3], sorted_square[0], sorted_square[2])
        return squareZ

    @staticmethod
    def find_squareZs(squares, edges_dict):
        """find_squareZs find non-repeating squareZs from square list and edges_dict

        Parameters
        ----------
        square : list
            list of squares in the format of [(1,2,3,4),(2,3,4,5),...]
        edges_dict : dict
            edges_dict in the format of {(1,2):1,(2,3):1,...}

        Returns
        -------
        squareZs : list
            list of squareZs in the format of [(1,2,3,4,1,3),(2,3,4,5,2,4),...]
        """
        squareZs = []
        for square in squares:
            a, b, c, d = square
            if (a, c) in edges_dict:
                # start with the smaller one
                squareZ = Motif.organize_squareZ(square, (a, c))
                squareZs.append(squareZ)

            if (b, d) in edges_dict:
                squareZ = Motif.organize_squareZ(square, (b, d))
                squareZs.append(squareZ)
        return squareZs

    @staticmethod
    def classify_squareZs(squareZs: tuple, edges_dict_double: dict):
        """
        classify_squareZs(squareZs:tuple,edges_dict_double:dict) -> dict
        Classify the squareZs into 32 catagoties.
        -------
        Parameters:
        squareZs: list of tuple
            a list of squareZs in the format of [(1,2,3,4,1,3),(2,3,4,5,2,4),...]
        edges_dict_double: dict
            edges_dict in the format of {(1,2):1,(2,3):1,...}
        -------
        Returns:
        sign_count: dict
            a dictionary of the counts of each category
        """
        all_possible_signs = ['----+', '-+--+', '+---+', '---++', '--+-+', '++--+', '+-+-+', '--+++', '-+-++', '-++-+', '+--++', '+-+++', '-++++', '+++-+', '++-++',
                              '+++++', '-----', '+----', '-+---', '--+--', '---+-', '+--+-', '+-+--', '++---', '-+-+-', '--++-', '-++--', '+++--', '++-+-', '-+++-', '+-++-', '++++-']
        sign_count = dict.fromkeys(all_possible_signs, 0)
        for squareZ in squareZs:
            signs = [Helper.get_sign(edges_dict_double[(squareZ[i], squareZ[i+1])])
                     for i in range(5)]
            joined_signs = ''.join(signs)
            sign_count[joined_signs] += 1
        return sign_count

    @staticmethod
    def find_and_classify_squareZs(squares: list, edges_dict_double: dict, graphletMode: bool = False):
        """
        find_and_classify_squareZs(squares:list, edges_dict_double:dict) -> dict
        Find and classify the squareZs into 32 catagoties.
        -------
        Parameters:
        squares: list
            a list of squares in the format of [(1,2,3,4),(2,3,4,5),...]
        edges_dict_double: dict
            edges_dict in the format of {(1,2):1,(2,1):1,...}
        -------
        Returns:
        sign_count: dict
            a dictionary of the counts of each category
        """
        # initial the result dict
        all_possible_signs = ['----+', '-+--+', '+---+', '---++', '--+-+', '++--+', '+-+-+', '--+++', '-+-++', '-++-+', '+--++', '+-+++', '-++++', '+++-+', '++-++',
                              '+++++', '-----', '+----', '-+---', '--+--', '---+-', '+--+-', '+-+--', '++---', '-+-+-', '--++-', '-++--', '+++--', '++-+-', '-+++-', '+-++-', '++++-']
        sign_count = dict.fromkeys(all_possible_signs, 0)

        for square in squares:
            a, b, c, d = square
            if (a, c) in edges_dict_double:
                # --graphletMode: skip if square has two diagonal edges
                if graphletMode and (b, d) in edges_dict_double:
                    continue
                # find the squareZ
                squareZ = Motif.organize_squareZ(square, (a, c))
                # classify the squareZ
                signs = [Helper.get_sign(edges_dict_double[(squareZ[i], squareZ[i+1])])
                         for i in range(5)]
                joined_signs = ''.join(signs)
                sign_count[joined_signs] += 1

            if (b, d) in edges_dict_double:
                # --graphletMode: skip if square has two diagonal edges
                if graphletMode and (a, c) in edges_dict_double:
                    continue
                # find the squareZ
                squareZ = Motif.organize_squareZ(square, (b, d))
                # classify the squareZ
                signs = [Helper.get_sign(edges_dict_double[(squareZ[i], squareZ[i+1])])
                         for i in range(5)]
                joined_signs = ''.join(signs)
                sign_count[joined_signs] += 1

        return sign_count

    @staticmethod
    def find_and_classify_squareXs(squares: list, edges_dict: dict):
        """
        find_and_classify_squareXs(squares:list, edges_dict_double:dict) -> dict
        Find and classify the squareXs into 64 catagoties.
        -------
        Parameters:
        squares: list
            a list of squares in the format of [(1,2,3,4),(2,3,4,5),...]
        edges_dict: dict
            edges_dict in the format of {(1,2):1,(2,3):-1,...}
        -------
        Returns:
        sign_count: dict
            a dictionary of the counts of each category
        """
        edges_dict_double = Helper.get_edges_dict_double(edges_dict)
        # initial the result dict
        all_possible_signs = ['------',
                              '--+---', '-+----', '----+-', '-----+', '+-----', '---+--',
                              '--+--+', '--++--', '-+-+--', '-+---+', '+----+', '--+-+-', '-++---', '---++-', '+-+---', '++----', '---+-+', '+---+-', '----++', '-+--+-', '+--+--',
                              '--+-++', '+++---', '++-+--', '---+++', '++---+', '-++--+', '+---++', '-+--++', '-++-+-', '+--++-', '--++-+', '+-+--+', '+-+-+-', '-+-+-+', '++--+-', '-+-++-', '+-++--', '+--+-+', '--+++-', '-+++--',
                              '+++-+-', '+-+++-', '-++-++', '+-+-++', '-+++-+', '++++--', '+-++-+', '++--++', '-+-+++', '-++++-', '++-++-', '+--+++', '+++--+', '--++++', '++-+-+',
                              '+-++++', '+++-++', '-+++++', '++-+++', '+++++-', '++++-+',
                              '++++++']

        sign_count = dict.fromkeys(all_possible_signs, 0)

        for square in squares:  # for each square, only 1 squareX will be found
            a, b, c, d = square
            if (a, c) in edges_dict_double and (b, d) in edges_dict_double:
                edges = [(a, b), (b, c), (c, d), (d, a), (a, c), (b, d)]
                signs = [Helper.get_sign(edges_dict_double[edge])
                         for edge in edges]
                joined_signs = ''.join(signs)
                sign_count[joined_signs] += 1
        return sign_count

    @staticmethod
    def count_motifs(edges_dict: dict, mode: str = 'triangle', exclude_self_loops: bool = True, graphletMode: bool = False):
        """count_motifs(edges_dict, mode='triangle'): count the number of motifs in the graph
        Parameters
        ----------
        edges_dict : dict
            dictionary of edges, example: {(1,2):1,(2,3):-1}
        mode : str
            'triangle','square', 'squareZ', 'squareX' , default 'triangle'
        exclude_self_loops : bool
            whether exclude self loops, default True
        graphletMode: bool
            whether in graphlet mode, default False. If True, square will exclude square with diagonal edges, squareZ will exclude squareZ with two diagonal edges.

        Returns
        -------
        res : dict
            dictionary of motifs count, example: {0: 0, 1: 0, 2: 0, 3: 0}
        """
        edges_dict_double = Helper.get_edges_dict_double(edges_dict)
        if mode == 'triangle':
            triangles = Motif.find_triangles(
                edges_dict, exclude_self_loops=exclude_self_loops)
            res = Motif.classify_triangles(triangles, edges_dict)
        elif mode == 'square':
            squares = Motif.find_squares(
                edges_dict, exclude_self_loops=exclude_self_loops)
            res = Motif.classify_squares(
                squares, edges_dict, graphletMode=graphletMode)
        elif mode == 'squareZ':
            squares = Motif.find_squares(
                edges_dict, exclude_self_loops=exclude_self_loops)
            res = Motif.find_and_classify_squareZs(
                squares, edges_dict_double, graphletMode=graphletMode)
        elif mode == 'squareX':
            squares = Motif.find_squares(
                edges_dict, exclude_self_loops=exclude_self_loops)
            res = Motif.find_and_classify_squareXs(squares, edges_dict)
        else:
            raise ValueError(
                'mode should be either triangle,square, squareZ or squareX')
        return res


class Randomization:
    @staticmethod
    # get probs
    def subnetwork_get_probs(edges_dict, subnetwork: str = 'neg', max_iters: int = 10000, stop_criterion: float = 10**(-3), verbose: bool = False):
        """subnetwork_get_probs 
        get the probs of links in edges_dict according to the degree sequence in pos or neg subnetwork.

        Parameters
        ----------
        edges_dict : dict
            dictionary of edges, example: {(1,2):1,(2,3):-1}
        subnetwork : str, optional
            Denote either randomize the neg subnetwork or pos subnetwork , by default 'neg'
        max_iters : int, optional
            The maximum iterations to update alphas, by default 10000
        stop_criterion : float, optional
            The stop criteria for updating alphas, by default 10**(-3)

        Returns
        -------
        probs : list
            the probs of links in G according to the degree sequence in subnetwork.
        """
        # divide the graph into pos,neg subgraphs
        edges_pos = []
        edges_neg = []
        for e in edges_dict:
            if edges_dict[e] > 0:
                edges_pos.append(e)
            else:
                edges_neg.append(e)
        edges = edges_pos + edges_neg

        edges_adj = calibration.formatting.edgelist_to_neighborhood(edges)
        edges_pos_adj = calibration.formatting.edgelist_to_neighborhood(
            edges_pos)
        edges_neg_adj = calibration.formatting.edgelist_to_neighborhood(
            edges_neg)

        if subnetwork == 'neg':
            rand_subnetwork = edges_neg
            rand_subnetwork_adj = edges_neg_adj
        else:
            rand_subnetwork = edges_pos
            rand_subnetwork_adj = edges_pos_adj

        # randomize either positive or negative subnetwork
        alphas, _, cur_iter = calibration.random_subnetwork.optimize_alpha_with_stop(
            edges_adj, rand_subnetwork_adj, max_iters=max_iters, stopping_criterion=stop_criterion)
        if verbose:
            print("cur_iter: ", cur_iter)

        probs = calibration.random_subnetwork.cal_probability(
            edges, rand_subnetwork, alphas=alphas)
        return probs

    @staticmethod
    def subnetwork_graph_from_prob(probs, sign: int = 1):
        """rand_graph_from_prob generate a random graph from a given probability dict

        Parameters
        ----------
        probs : dict
            probability in the form of a dict, for example: {('a','b'):0.5,('a','c'):0.3,('b','c'):0.1}
        sign : int, optional
            sign of the edges, by default 1

        Returns
        -------
        dict
            a dict of edges, for example: {('a','b'):1,('a','c'):-1,('b','c'):1}
        """
        edges = {}
        for edge in probs:
            prob = probs[edge]
            if prob > random.random():
                edges[edge] = sign
        return edges

    @staticmethod
    def rand_subnetwork(edges_dict: dict, probs: dict, subnetwork: str = 'neg'):
        """rand_subnetwork randomize the network by randomizing the subnetwork

        Parameters
        ----------
        edges_dict : dict
            dictionary of edges, example: {(1,2):1,(2,3):-1}
        probs : dict
            the probs of links in edges_dict according to the degree sequence in pos or sub subnetwork.
        subnetwork : str, optional
            Denote either randomize the neg subnetwork or pos subnetwork , by default 'neg'. Note it has to be the same as the probs.

        Returns
        -------
        edges_dict_rand: dict
            dictionary of edges after randomization, with signs
        """
        if subnetwork == 'neg':
            # get the neg subnetwork by probs
            edges_neg_rand = Randomization.subnetwork_graph_from_prob(
                probs, sign=-1)
            # set the remaining links as pos subnetwork
            edges_pos_rand = {
                e: 1 for e in edges_dict if e not in edges_neg_rand}

        else:
            # get the pos subnetwork by probs
            edges_pos_rand = Randomization.subnetwork_graph_from_prob(
                probs, sign=1)
            # set the remaining links as neg subnetwork
            edges_neg_rand = {
                e: -1 for e in edges_dict if e not in edges_pos_rand}

        # construct the rand edges_dict
        edges_dict_rand = {**edges_pos_rand, **
                           edges_neg_rand}  # merge two dicts

        return edges_dict_rand



class Helper:
    @staticmethod
    def df_keep_unique_squareZ(df: pd.DataFrame):
        """df_keep_unique_squareZ keep only unique squareZs in a dataframe
        There are 32 possible squareZs, but only 20 unique ones. This function keeps only the unique ones in a dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            a dataframe of squareZs

        Returns
        -------
        df : pd.DataFrame
            a dataframe of unique squareZs
        """
        # unique_signs = {'----+': [], '-+--+': ['---++'], '+---+': ['--+-+'], '++--+': ['--+++'], '+-+-+': [], '-+-++': [], '-++-+': ['+--++'], '+-+++': ['+++-+'], '-++++': ['++-++'], '+++++': [],
        #                 '-----': [], '+----': ['--+--'], '-+---': ['---+-'], '+--+-': ['-++--'], '+-+--': [], '++---': ['--++-'], '-+-+-': [], '+++--': ['+-++-'], '++-+-': ['-+++-'], '++++-': []}
        unique_signs = {'----+': [],
                        '-----': [],
                        '-+--+': ['---++', '+---+', '--+-+'],
                        '-+---': ['--+--', '+----', '---+-'],
                        '++--+': ['--+++'],
                        '++---': ['--++-'],
                        '-++-+': ['+--++'],
                        '-++--': ['+--+-'],
                        '+-+-+': ['-+-++'],
                        '+-+--': ['-+-+-'],
                        '+-+++': ['+++-+', '-++++', '++-++'],
                        '+-++-': ['+++--', '++-+-', '-+++-'],
                        '+++++': [],
                        '++++-': []}
        df1 = df.copy()
        for unique_sign in unique_signs:
            for iso in unique_signs[unique_sign]:
                df1[unique_sign] = df1[unique_sign] + df1[iso]
                df1.drop(iso, axis=1, inplace=True)
        unique_signs_name = list(unique_signs.keys())
        df1 = df1.reindex(unique_signs_name, axis=1)
        return df1

    @staticmethod
    def df_keep_unique_squareX(df: pd.DataFrame):
        """df_keep_unique_squareX keep only unique squareXs in a dataframe
        There are 64 possible squareXs, but only 11 unique ones. This function keeps only the unique ones in a dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            a dataframe of squareXs

        Returns
        -------
        df : pd.DataFrame
            a dataframe of unique squareXs
        """
        unique_signs = {'------': [],
                        '--+---': ['-+----', '----+-', '-----+', '+-----', '---+--'],
                        '--+--+': ['--++--', '-+---+', '+----+', '--+-+-', '-++---', '---++-', '++----', '---+-+', '+---+-', '-+--+-', '+--+--'],
                        '-+-+--': ['+-+---', '----++'],
                        '--+-++': ['+++---', '++-+--', '---+++', '+---++', '-+--++', '+-+--+', '+-+-+-', '-+-+-+', '-+-++-', '+-++--', '-+++--'],
                        '++---+': ['-++-+-', '+--++-', '--++-+'],
                        '-++--+': ['++--+-', '+--+-+', '--+++-'],
                        '+++-+-': ['+-+++-', '-++-++', '-+++-+', '+-++-+', '++--++', '-++++-', '++-++-', '+--+++', '+++--+', '--++++', '++-+-+'],
                        '+-+-++': ['++++--', '-+-+++'],
                        '+-++++': ['+++-++', '-+++++', '++-+++', '+++++-', '++++-+'],
                        '++++++': []}
        df1 = df.copy()
        for unique_sign in unique_signs:
            for iso in unique_signs[unique_sign]:
                df1[unique_sign] = df1[unique_sign] + df1[iso]
                df1.drop(iso, axis=1, inplace=True)
        unique_signs_name = list(unique_signs.keys())
        df1 = df1.reindex(unique_signs_name, axis=1)
        return df1

    @staticmethod
    def get_sign(num: int or float):
        """get_sign(num:int or float) -> str
        Get the sign of a number.
        -------
        Parameters:
        num: int or float
            a number
        -------
        Returns:
        sign: str
            the sign of the number, either '+' or '-' or '0'
        """
        if num > 0:
            return '+'
        elif num < 0:
            return '-'
        else:
            return '0'

    def get_dict_vals(dic: dict, keys: list):
        """get_dict_vals(dic:dict, keys:list) -> list
        Get the values of a dictionary given the keys. If the key is not in the dictionary, return 0.
        -------
        Parameters:
        dic: dict
            a dictionary
        keys: list
            a list of keys
        -------
        Returns:
        vals: list
            a list of values
        """
        vals = []
        for key in keys:
            vals.append(dic.get(key, 0))
        return vals

    def get_edges_dict_double(edges_dict: dict):
        """get_edges_dict_double(edges_dict:dict) -> dict
        Get the double edges dictionary from a single edges dictionary.
        -------
        Parameters:
        edges_dict: dict
            a single edges dictionary
        -------
        Returns:
        double_edges_dict: dict
            a double edges dictionary
        """
        double_edges_dict = {}
        for edge in edges_dict:
            double_edges_dict[edge] = edges_dict[edge]
            double_edges_dict[(edge[1], edge[0])] = edges_dict[edge]
        return double_edges_dict

    @staticmethod
    def load_data(dataset: str, exclude_self_loops: bool = True, lcc: bool = True, directed: bool = False):
        """load_data load the dataset

        Available datasets: 
        'Congress','Bitcoin-Alpha','WikiElec','Slashdot','Epinions','GIEE','Drug','Drug_down','Pardus_sample'.

        Parameters
        ----------
        dataset : str
            the dataset to load
        exclude_self_loops : bool, optional
            whether to exclude self loops, by default True
        lcc : bool, optional
            whether to keep the largest connected component, by default True

        Returns
        -------
        edges_dict : dict
            dictionary of edges, example: {(1,2):1,(2,3):-1}
        """

        df = pd.read_csv('datasets/processed_lcc_' + dataset + '.csv')
        edges_dict, name2id, id2name = Formatter.df_to_edgedict(df, 'node1', 'node2', 'sign' ,nodeMapping=False, exclude_self_loops=exclude_self_loops, directed=directed)
        
        # keep the largest connected component
        if lcc:
            sys.setrecursionlimit(1000000)
            edges_dict = GraphAnalysis.get_lcc_edges(edges_dict)

        return edges_dict, name2id, id2name

    @staticmethod
    def get_node_degree(edges: list, node: list):
        """get_node_degree get the degree of a series of node

        Parameters
        ----------
        edges : list
            list of edges
        node : list
            list of nodes

        Returns
        -------
        degree : list
            list of degree in the same order as the node list
        """
        adj = defaultdict(set)
        for edge in edges:
            adj[edge[0]].add(edge[1])
            adj[edge[1]].add(edge[0])
        degree = [len(adj[node]) for node in node]
        return degree

    # check if the signed degree is preserved
    @staticmethod
    def check_signed_degree(edges_dict, n_sample=100, randomization: str = 'signed_rewire', show: str = 'signed_degree', signed_rewire_incompatible: str = 'random'):

        edges_pos = [edge for edge in edges_dict if edges_dict[edge] > 0]
        edges_neg = [edge for edge in edges_dict if edges_dict[edge] < 0]
        nodes = list(set([edge[0] for edge in edges_dict] +
                     [edge[1] for edge in edges_dict]))
        pos_degree = Helper.get_node_degree(edges_pos, nodes)
        neg_degree = Helper.get_node_degree(edges_neg, nodes)

        pos_degrees_ = []
        neg_degrees_ = []

        for i in range(n_sample):
            if randomization == 'rewire':
                edges_dict_rand = Randomization.rand_rewire(
                    edges_dict, iters=-1)
            elif randomization == 'signed_rewire':
                edges_dict_rand = Randomization.rand_signed_rewire(
                    edges_dict, iters=-1, incompatibleEdges=signed_rewire_incompatible)
            elif randomization == 'sign_shuffle':
                edges_dict_rand = Randomization.rand_sign_shuffle(edges_dict)
            elif randomization == 'bernoulli':
                edges_dict_rand = Randomization.rand_bernoulli(edges_dict)
            elif randomization == 'subnetwork':
                if i == 0:
                    probs = Randomization.subnetwork_get_probs(
                        edges_dict, subnetwork='neg', max_iters=10000, stop_criterion=1e-3)
                edges_dict_rand = Randomization.rand_subnetwork(
                    edges_dict, probs, subnetwork='neg')

            edges_pos_rand = [
                edge for edge in edges_dict_rand if edges_dict_rand[edge] > 0]
            edges_neg_rand = [
                edge for edge in edges_dict_rand if edges_dict_rand[edge] < 0]
            pos_degree_rand = Helper.get_node_degree(edges_pos_rand, nodes)
            neg_degree_rand = Helper.get_node_degree(edges_neg_rand, nodes)
            pos_degrees_.append(pos_degree_rand)
            neg_degrees_.append(neg_degree_rand)

        pos_degrees_mean = np.mean(pos_degrees_, axis=0)
        neg_degrees_mean = np.mean(neg_degrees_, axis=0)

        if show == 'signed_degree':
            # plot two subplots
            fig, axs = plt.subplots(1, 2, figsize=(10, 5), dpi=150)

            # set title for both subplots
            fig.suptitle(randomization + ', sample = ' +
                         str(n_sample), fontsize=16)

            lim = max(pos_degree) * 1.1
            axs[0].set_xlim(0, lim)
            axs[0].set_ylim(0, lim)
            axs[0].plot([0, lim], [0, lim], 'k--')
            axs[0].scatter(pos_degree, pos_degrees_mean,
                           label='original', alpha=1, s=15)

            lim = max(neg_degree) * 1.1
            axs[1].set_xlim(0, lim)
            axs[1].set_ylim(0, lim)
            axs[1].plot([0, lim], [0, lim], 'k--', label='y=x')
            axs[1].scatter(neg_degree, neg_degrees_mean,
                           label='original', alpha=1, s=15)

            # set xlabel and ylabel
            axs[0].set_xlabel('positive degree')
            axs[0].set_ylabel('positive degree after randomization')
            axs[1].set_xlabel('negative degree')
            axs[1].set_ylabel('negative degree after randomization')
            plt.show()

        # plot the average degree
        elif show == 'total_degree':
            fig, axs = plt.subplots(1, 1, figsize=(5, 5), dpi=150)
            total_degree = np.array(pos_degree) + np.array(neg_degree)
            total_degree_rand = np.array(
                pos_degrees_mean) + np.array(neg_degrees_mean)
            lim = max(total_degree) * 1.1
            axs.set_xlim(0, lim)
            axs.set_ylim(0, lim)
            axs.plot([0, lim], [0, lim], 'k--', label='y=x')
            axs.scatter(total_degree, total_degree_rand, s=15)
            plt.xlabel('total degree')
            plt.ylabel('total degree after randomization')
            plt.title('total degree, ' + randomization +
                      ', sample = ' + str(n_sample))
            plt.show()

    @staticmethod
    def check_file_path(file_path: str):
        """check_file_path check if the file path exists, if not, create it

        Parameters
        ----------
        file_path : str
            file path to check

        Returns
        -------
        None
        """
        if not os.path.exists(file_path):
            os.makedirs(file_path)

    @staticmethod
    def read_edges_dict_from_file(filepath: str):
        # if file exsit, open it with np, otherwise return None
        try:
            edges_dict = np.load(filepath, allow_pickle=True).item()
        except:
            edges_dict = None
        return edges_dict

    @staticmethod
    def save_edges_dict_to_file(edges_dict: dict, filepath: str):
        np.save(filepath, edges_dict)

    @staticmethod
    def get_adj_list(edges_dict):
        """get_adj_list returns the adjacency list of the graph

        Parameters
        ----------
        edges_dict : dict
            a dictionary of edges

        Returns
        -------
        dict
            a dictionary of adjacency list, for example, {1: {2, 3}, 2: {1, 3}, 3: {1, 2}}.
        """
        adj = defaultdict(set)
        for edge in edges_dict:
            adj[edge[0]].add(edge[1])
            adj[edge[1]].add(edge[0])
        return adj

    @staticmethod
    def dataset_overview(df, dataset: str):
        """dataset_overview the overview of the dataset

        Parameters
        ----------
        df : pd.DataFrame
            dataframe of the dataset with columns ['id1','id2','sign']
        dataset : str
            name of the dataset

        Returns
        -------
        pd.DataFrame
            dataframe of the overview
        """
        results = []
        # directed graph
        all_edges = defaultdict(list)
        incompatible_edges = {}
        compatible_edges = {}

        for row in df.values:
            source, target, sign = row
            all_edges[(source, target)].append(sign)
        for edge in all_edges:
            # if the edges have different signs, then it is an incompatible edge
            if -1 in all_edges[edge] and 1 in all_edges[edge]:
                incompatible_edges[edge] = all_edges[edge]
            # if the edges have the same sign, then it is a compatible edge
            else:
                compatible_edges[edge] = all_edges[edge][0]

        n_directed_edges = len(all_edges)
        n_directed_nodes = len(
            set([x for x, y in all_edges.keys()]+[y for x, y in all_edges.keys()]))
        n_directed_incompatible_edges = len(incompatible_edges)
        n_directed_incompatible_nodes = len(set(
            [x for x, y in incompatible_edges.keys()]+[y for x, y in incompatible_edges.keys()]))
        n_directed_compatible_edges = len(compatible_edges)
        n_directed_compatible_nodes = len(set(
            [x for x, y in compatible_edges.keys()]+[y for x, y in compatible_edges.keys()]))

        # undirected graph
        all_edges = defaultdict(list)
        incompatible_edges = {}
        compatible_edges = {}

        for row in df.values:
            source, target, sign = row
            edge = (source, target) if source < target else (target, source)
            all_edges[edge].append(sign)

        for edge in all_edges:
            # if the edges have different signs, then it is an incompatible edge
            if -1 in all_edges[edge] and 1 in all_edges[edge]:
                incompatible_edges[edge] = all_edges[edge]
            # if the edges have the same sign, then it is a compatible edge
            else:
                compatible_edges[edge] = all_edges[edge][0]

        self_loops = {}
        for edge in compatible_edges:
            if edge[0] == edge[1]:
                self_loops[edge] = compatible_edges[edge]

        compatible_edges_without_self = {
            k: v for k, v in compatible_edges.items() if k not in self_loops}
        compatible_nodes_without_self = set([x for x, y in compatible_edges_without_self.keys(
        )]+[y for x, y in compatible_edges_without_self.keys()])
        pos_egdes = {k: v for k,
                     v in compatible_edges_without_self.items() if v > 0}
        n_pos_edges = len(pos_egdes)
        pos_egdes_ratio = round(
            len(pos_egdes)/len(compatible_edges_without_self), 4)
        # round to 2 decimal places
        density = round(len(compatible_edges_without_self)/(
            len(compatible_nodes_without_self)*(len(compatible_nodes_without_self)-1)/2), 4)

        n_undirected_edges = len(all_edges)
        n_undirected_nodes = len(
            set([x for x, y in all_edges.keys()]+[y for x, y in all_edges.keys()]))
        n_undirected_incompatible_edges = len(incompatible_edges)
        n_undirected_incompatible_nodes = len(set(
            [x for x, y in incompatible_edges.keys()]+[y for x, y in incompatible_edges.keys()]))
        n_undirected_compatible_edges = len(compatible_edges)
        n_undirected_compatible_nodes = len(set(
            [x for x, y in compatible_edges.keys()]+[y for x, y in compatible_edges.keys()]))
        n_undirected_self_loops = len(self_loops)
        undirected_incompatible_edges_ratio = round(
            n_undirected_incompatible_edges/n_undirected_edges, 4)

        # LCC
        final_edges = [[edge[0], edge[1], compatible_edges_without_self[edge]]
                       for edge in compatible_edges_without_self]
        df = pd.DataFrame(final_edges, columns=['source', 'target', 'sign'])
        G = nx.from_pandas_edgelist(df, 'source', 'target', 'sign')
        lcc_nodes = max(nx.connected_components(G), key=len)
        n_lcc_nodes = len(lcc_nodes)
        lcc_G = G.subgraph(lcc_nodes).copy()
        n_lcc_edges = lcc_G.number_of_edges()

        res = [dataset, n_directed_nodes, n_directed_edges, n_directed_compatible_nodes, n_directed_compatible_edges, n_directed_incompatible_nodes, n_directed_incompatible_edges, n_undirected_nodes, n_undirected_edges, n_undirected_compatible_nodes,
               n_undirected_compatible_edges, n_undirected_incompatible_nodes, n_undirected_incompatible_edges, undirected_incompatible_edges_ratio, n_undirected_self_loops, n_pos_edges, pos_egdes_ratio, density, n_lcc_nodes, n_lcc_edges]
        results.append(res)

        data_names = ['Dataset', 'n_directed_nodes', 'n_directed_edges', 'n_directed_compatible_nodes', 'n_directed_compatible_edges', 'n_directed_incompatible_nodes', 'n_directed_incompatible_edges', 'n_undirected_nodes', 'n_undirected_edges',
                      'n_undirected_compatible_nodes', 'n_undirected_compatible_edges', 'n_undirected_incompatible_nodes', 'n_undirected_incompatible_edges', 'undirected_incompatible_edges_ratio', 'n_undirected_self_loops', 'n_pos_edges', 'pos_ratio', 'density', 'n_lcc_nodes', 'n_lcc_edges']
        df_res = pd.DataFrame(results[0])
        # set the index for the dataframe
        df_res.index = data_names

        return df_res


class GraphAnalysis:
    @staticmethod
    def get_connected_component(edges_dict):
        """get_connected_component returns the connected components of the graph

        Parameters
        ----------
        edges_dict : dict
            a dictionary of edges, for example {(1,2): 1, (1,3): 1, (2,3): -1}

        Returns
        -------
        list
            a list of nodes in connected components, for example, [{1, 2, 3}, {4, 5, 6}]
        """
        def DFS(temp, node, visited, adj):
            # Mark the current vertex as visited
            visited[node] = True

            # Store the vertex to list
            temp.add(node)
            # Repeat for all vertices adjacent
            # to this vertex v
            for i in adj[node]:
                if visited[i] == False:
                    # Update the list
                    temp = DFS(temp, i, visited, adj)
            return temp

        adj = Helper.get_adj_list(edges_dict)
        nodes = set([e[0] for e in edges_dict]+[e[1] for e in edges_dict])
        visited = {}
        cc = []
        for node in nodes:
            visited[node] = False
        for node in nodes:
            if visited[node] == False:
                temp = set()
                cc.append(DFS(temp, node, visited, adj))
        return cc

    @staticmethod
    def get_lcc_edges(edges_dict):
        """get_lcc_edges returns the edges of the largest connected component

        Parameters
        ----------
        edges_dict : dict
            a dictionary of edges, for example {(1,2): 1, (1,3): 1, (2,3): -1}

        Returns
        -------
        edges_dict_lcc : dict
            a dictionary of edges in the largest connected component, for example {(1,2): 1, (1,3): 1, (2,3): -1}
        """
        cc = GraphAnalysis.get_connected_component(edges_dict)
        lcc_nodes = max(cc, key=len)
        edges_dict_lcc = {
            e: edges_dict[e] for e in edges_dict if e[0] in lcc_nodes and e[1] in lcc_nodes}
        return edges_dict_lcc

