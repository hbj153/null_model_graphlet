import math
from itertools import combinations
from collections import defaultdict
import networkx as nx
import random
import time
import copy
import numpy as np
import matplotlib.pyplot as plt

def get_common_nodes(E1,E2):
    set1 = set(np.array(E1).flatten())
    set2 = set(np.array(E2).flatten())
    commonNodes = list(set1.intersection(set2))
    return commonNodes
def count_unique_nodes(E1):
    return len(set(np.array(E1).flatten()))

def pipeline_check_degree_dismatch(E1,E2,name1="E1",name2="E2",xlim=None,ylim=None,logscale=False,dpi=200):
    nodes = get_common_nodes(E1,E2)
    E1dict = formatting.edgelist_to_neighborhood(E1)
    E2dict = formatting.edgelist_to_neighborhood(E2)
    E1_degrees = random_subnetwork.cal_node_degree(E1dict)
    E2_degrees = random_subnetwork.cal_node_degree(E2dict)
    print("unqiue nodes in "+name1+" = %s"%(count_unique_nodes(E1)))
    print("unqiue nodes in "+name2+" = %s"%(count_unique_nodes(E2)))
    print("common nodes in "+name1+" and "+name2+" = %s"%len(nodes))
    plt.figure(dpi=dpi,figsize=(5,5))
    
    x = np.log10(dict_values(E1_degrees,nodes)) if logscale else dict_values(E1_degrees,nodes)
    y = np.log10(dict_values(E2_degrees,nodes)) if logscale else dict_values(E2_degrees,nodes)
    plt.scatter(x,y,alpha=0.5,c = "#BEB8DC")
    
    if logscale:
        plt.xlabel("log10(degrees) in "+name1)
        plt.ylabel("log10(degrees) in "+name2)
    else:
        plt.xlabel("degrees in "+name1)
        plt.ylabel("degrees in "+name2)
    lim = max(max(x),max(y))
    xlim = lim if not xlim else xlim
    ylim = lim if not ylim else ylim
    
    plt.xlim(0,xlim)
    plt.ylim(0,ylim)
    plt.show()

def find_selfNodes(a1dict):
    '''
    Find nodes that have self-interations in a given dictionary.
    '''
    selfNodes = []
    for node in a1dict:
        if node in a1dict[node]:
            selfNodes.append(node)
    return selfNodes

def dict_remove_self(dic):
    '''
    Remove the key value in dic[key].
    '''
    dic1 = copy.deepcopy(dic)
    todeletes = []
    for node in dic1:
        if node in dic1[node]:
            dic1[node].remove(node)
        if not dic1[node]: #如果去电self之后为空集，则删除该点
            todeletes.append(node)
    for todelete in todeletes:
        del dic1[todelete]
    return dic1

def dict_values(dict1,keys):
    '''
    dict_values(dict1,keys)

    Return the values by the order of a given key list.
    
    Parameters
    ----------
    dict1: Dictionary.
    keys: a list of keys.
    
    Returns
    -------
    dict_value: The values of dictionart following the order of given keys.
    '''
    degrees = []
    for cur_key in keys:
        if cur_key in dict1.keys():
            degrees.append(dict1[cur_key])
        else:
            degrees.append(0)
    #return [dict1[cur_key] for cur_key in keys]
    return degrees

def count_overlap(elist1,elist2):
    return len(set(elist1).intersection(elist2))

def sort_elist(elist,strip=True,selfLink=False):
    '''
    strip: if True, the gene name will change from "YAL034W-A" to "YAL034WA"
    selfLink: if False, self-interactions will be removed from the edgelist.
    '''
    if len(elist)==0 or type(elist[0][0])!=str: return list(set(sorted([tuple(sorted(x)) for x in elist])))

    if strip==True:
        res = list(set(sorted([tuple(sorted([x[0].replace("-",""),x[1].replace("-","")])) for x in elist])))
        #res = list(set(sorted([tuple(sorted([x[0].split("-")[0],x[1].split("-")[0]])) for x in elist])))
    else:
        res = list(set(sorted([tuple(sorted(x)) for x in elist])))
        
    if selfLink==False:
        res = [pair for pair in res if pair[0]!=pair[1]]
    
    return res

def list_sub(A,B):
    return [x for x in A if x not in B]

# normalized score


def cal_score(ms, bs, bes, cs, ces):
    '''
    f = (m-b)/(c-b)
    m is constant
    '''
    scores = []
    score_sigmas = []

    # if pass integers
    if type(ms) == int:
        ms, bs, bes, cs, ces = [ms], [bs], [bes], [cs], [ces]

    # if pass arrays
    for m, b, be, c, ce in zip(ms, bs, bes, cs, ces):
        if m != 0:  # if obs != 0
            if c-b != 0:
                score = (m-b)/(c-b)
                score_sigma = (1/(c-b)**2)*np.sqrt((m-c)
                                                   ** 2*be**2+(b-c)**2*ce**2)
            else:
                score = None
                score_sigma = None
        else:
            score = 0
            score_sigma = 0
        scores.append(score)
        score_sigmas.append(score_sigma)

    return scores, score_sigmas

def get_maxOverlap(elist1,elist2,method="MinDeg"):
    '''
    get_maxOverlap(elist1,elist2)

    Return the possible maxOverlap between two networks. 
    
    Parameters
    ----------
    elist1: network 1 in edgelist format.
    elist2: network 2 in edgelist format.
    method: "MinDeg" or "ComNode".
            - "MinDeg": The two networks are first constrained to subnetworks with common nodes. The maxOverlap is defined by summing up minimum degree of each node in the two networks and divide by 2.
    
    Returns
    -------
    maxOverlap: The possible maxOverlap between two given networks.
    '''
    '''
    if method == "MinDeg":
        dict1 = formatting.edgelist_to_neighborhood(elist1)
        dict2 = formatting.edgelist_to_neighborhood(elist2)
        degree1 = random_subnetwork.cal_node_degree(dict1)
        degree2 = random_subnetwork.cal_node_degree(dict2)
        commonNodes = get_common_nodes(elist1,elist2)
        degree = {}
        for node in commonNodes:
            degree[node] = min(degree1[node],degree2[node])
        maxOverlap = sum(degree.values())/2
    
    if method == "MinDeg":
        commonNodes = get_common_nodes(elist1,elist2)
        elist1_com = [pair for pair in elist1 if (pair[0] in commonNodes) and (pair[1] in commonNodes)]
        elist2_com = [pair for pair in elist2 if (pair[0] in commonNodes) and (pair[1] in commonNodes)]
        degree1 = random_subnetwork.cal_node_degree(elist1_com)
        degree2 = random_subnetwork.cal_node_degree(elist2_com)
        degree = {}
        for node in commonNodes:
            if node in degree1 and node in degree2:
                degree[node] = min(degree1[node],degree2[node])
            else:
                degree[node]=0
        maxOverlap = sum(degree.values())/2
    '''
    
    if method == "MinDeg": # keep the self-interactions
        commonNodes = get_common_nodes(elist1,elist2)
        elist1_com = [pair for pair in elist1 if (pair[0] in commonNodes) and (pair[1] in commonNodes)]
        elist2_com = [pair for pair in elist2 if (pair[0] in commonNodes) and (pair[1] in commonNodes)]
        elist1_noself = []
        elist1_self = []
        elist2_noself = []
        elist2_self = []
        for pair in elist1_com:
            if pair[0]==pair[1]:
                elist1_self.append(pair)
            else:
                elist1_noself.append(pair)
        for pair in elist2_com:
            if pair[0]==pair[1]:
                elist2_self.append(pair)
            else:
                elist2_noself.append(pair)        
        degree1 = random_subnetwork.cal_node_degree(elist1_noself)
        degree2 = random_subnetwork.cal_node_degree(elist2_noself)
        degree = {}
        for node in commonNodes:
            if node in degree1 and node in degree2:
                degree[node] = min(degree1[node],degree2[node])
            else:
                degree[node]=0
        nSelf = len(set(elist1_self).intersection(elist2_self))
        maxOverlap = sum(degree.values())/2 + nSelf
    
    if method == "ComNode":
        commonNodes = get_common_nodes(elist1,elist2)
        E1_filtered = [link for link in elist1 if (link[0] in commonNodes and link[1] in commonNodes)]
        E2_filtered = [link for link in elist2 if (link[0] in commonNodes and link[1] in commonNodes)]
        maxOverlap = min(len(E1_filtered),len(E2_filtered))
    
    return maxOverlap


class formatting:
    
    def edgelist_to_neighborhood(br):
        relation = defaultdict(set)
        for nodeArr in br:
            relation[nodeArr[0]].add(nodeArr[1])
            relation[nodeArr[1]].add(nodeArr[0])
        return relation

    def neighborhood_to_edgelist(N):
        '''
        neighborhood_to_edgelist(N)

        Convert node pairs in neighborhood format to edgelist format.

        Parameters
        ----------
        N: Input network in neighborhood formation. For example, N = {"A": {"B", "C"},"B":{"A"},"C":{"A"}}

        Returns
        -------
        edgelist: Output network in edgelist format like [("A","B"),("B","C")]

        '''
        edgelist = []
        for nodeArr in N.keys():
            for neighbor in N[nodeArr]:
                if (nodeArr,neighbor) not in edgelist and (neighbor,nodeArr) not in edgelist:
                    edgelist.append(tuple(sorted((nodeArr,neighbor))))
        return sorted(edgelist)
    
    def neighborhood_to_adjacency(N):
        '''
        neighborhood_to_adjacency(N)

        Convert network in neighborhood format to adjacency matrix format.

        Parameters
        ----------
        N: Input network in neighborhood formation. For example, N = {"A": {"B", "C"},"B":{"A"},"C":{"A"}}

        Returns
        -------
        nodelist: sorted nodelist corresponds to the adjacency matrix.
        A: Output network in adjacency matrix function.
        '''
        nodelist = sorted(N.keys())
        edgelist = formatting.neighborhood_to_edgelist(N)
        G = nx.from_edgelist(edgelist)
        A = nx.to_numpy_matrix(G,nodelist=nodelist)
        return nodelist,A
    
    def edgelist_to_adjacency(edgelist):
        '''
        edgelist_to_adjacency(edgelist)

        Convert network in edgelist format to adjacency matrix format.

        Parameters
        ----------
        N: Input network in edgelist formation.

        Returns
        -------
        nodelist: sorted nodelist corresponds to the adjacency matrix.
        A: Output network in adjacency matrix function.
        '''
        N = formatting.edgelist_to_neighborhood(edgelist)
        nodelist = sorted(N.keys())
        G = nx.from_edgelist(edgelist)
        A = nx.to_numpy_matrix(G,nodelist=nodelist)
        return nodelist,A
       
       
class random_subnetwork:

    def cal_node_degree(G):
        '''
        cal_node_degree(G)

        Calculate the node degree for a given network.

        Parameters
        ----------
        G: Network in neighborhood format like Gdict = {"A": {"B", "C"},"B":{"A"},"C":{"A"}} or edgelist format.

        Returns
        -------
        degrees: A dictionary with node degrees of each node. Example: {'A': 2, 'B': 1, 'C': 1}
        '''
        if type(G)==list:
            Gdict = formatting.edgelist_to_neighborhood(G)
        else:
            Gdict = G
        degrees = {}
        for i in Gdict:
            degrees[i]=len(Gdict[i])
        return degrees
    def optimize_alpha(G0dict0,G1dict0,iters=1000,probeNode=0):
        
        '''
        optimize_alpha(G0dict,G1dict,iters=1000,probeNode=0

        Optimize the alpha for nodes in G1dict.

        Parameters
        ----------
        G0dict: Complete network in neighborhood format. For example, N = {"A": {"B", "C"},"B":{"A"},"C":{"A"}}
        G1dict: Reference network in neighborhood format that provides the node degree constriants.
        iters: The number of iterations for updating alphas.
        probeNode: The index of the probe node. The alpha history will be returned for the probe node.

        Returns
        -------
        alphas: A dictionary contains the optimized alphas for each node.
        alpha_probe: A list contains the history alphas for the probleNode.
        '''
        G0dict = dict_remove_self(G0dict0) # remove self-interactions
        G1dict = dict_remove_self(G1dict0) # remove self-interactions
        degrees = random_subnetwork.cal_node_degree(G1dict) # reference degree sequence generated from G1
        alphas = {}.fromkeys(G1dict.keys(),1) # initialize alphas for all nodes to 1
        alpha_probe = []
        for itering in range(iters):
            alphas_tem = {}.fromkeys(G1dict.keys(),1) # save the new alphas, update alphas until the end of iteration.
            for i in G1dict:
                Sigma = 0
                for j in G0dict[i]: # only the links exsit in G0 will be counted.
                    if j in G1dict.keys(): # if node j also in G1
                        Sigma += 1/(alphas[j]+1/alphas[i])
                alphas_tem[i] = Sigma/degrees[i]
            alphas.update(alphas_tem)
            if probeNode != None:
                alpha_probe.append([itering,alphas[list(G1dict.keys())[probeNode]]])
            #if itering+10>iters: print(alphas)
        return alphas,alpha_probe


    def optimize_alpha_with_stop(G0dict0, G1dict0, max_iters=1000, stopping_criterion=-1):
        '''
        optimize_alpha_with_stop(G0dict0,G1dict0,max_iters=1000,stopping_criterion = -1):

        Optimize the alpha for nodes in G1dict.

        Parameters
        ----------
        G0dict: Complete network in neighborhood format. For example, N = {"A": {"B", "C"},"B":{"A"},"C":{"A"}}
        G1dict: Reference network in neighborhood format that provides the node degree constriants.
        iters: The number of iterations for updating alphas.
        probeNode: The index of the probe node. The alpha history will be returned for the probe node.

        Returns
        -------
        alphas: A dictionary contains the optimized alphas for each node.
        alpha_history: A list contains the alpha history for the probe node.
        cur_iter: The number of iterations when stop updating alphas.
        '''

        G0dict = dict_remove_self(G0dict0)  # remove self-interactions
        G1dict = dict_remove_self(G1dict0)  # remove self-interactions
        # reference degree sequence generated from G1
        degrees = random_subnetwork.cal_node_degree(G1dict)
        # initialize alphas for all nodes to 1
        alphas = {}.fromkeys(G1dict.keys(), 1)
        alphas_history = []
        rel_change = {}.fromkeys(G1dict.keys(), 999)
        alpha_probe = []
        for itering in range(max_iters):
            cur_iter = itering
            # if the maximum relative change of alphas is smaller than the stopping criterion, stop updating alphas.
            if max(rel_change.values()) < stopping_criterion:
                break
            # save the new alphas to alphas_tem, update alphas until the end of iteration.
            alphas_tem = {}.fromkeys(G1dict.keys(), 1)
            for i in G1dict:
                Sigma = 0
                for j in G0dict[i]:  # only the links exsit in G0 will be counted.
                    # if node j also in G1, otherwise degrees[j]=0
                    if j in G1dict.keys():
                        Sigma += 1 / (alphas[j] + 1 / alphas[i])
                alphas_tem[i] = Sigma / degrees[i]
                diff = (alphas_tem[i] - alphas[i])
                rel_change[i] = abs(diff / alphas[i])
            alphas.update(alphas_tem)
            alphas_history.append(list(alphas.values()))

        return alphas, alphas_history, cur_iter
    
    def cal_probability(G0elist,G1elist,alphas):

        '''
        cal_probability(G0elist,alphas)

        Calculate the probability for links in G0elist.

        Parameters
        ----------
        G0elist: The complete network in edgelist format. Example: G0elist = [('A', 'B'), ('A', 'C')]
        G1elist: Reference network in neighborhood format that provides the node degree constriants. Here it is used to determine whether a self-interaction si allowed.
                If (i,i) in G1elist, pii=1, else, pii=0.
        alphas: The optimized alphas for each node(of G1, the reference network).

        Returns
        -------
        probs: A dictionary contains the connection probability of the edges in G0elist.
        '''

        probs = {}
        for link in G0elist:
            if link[0]==link[1]: # self-interactions 
                if link in G1elist: # exsit in G1
                    probs[link] = 1
                else: # self-interaction not exsit in G1
                    probs[link] = 0
            else: # not self-interactions
                if link[0] in alphas.keys() and link[1] in alphas.keys(): # Only count links that both nodes in alpha dict
                    probs[link] = 1/(1+alphas[link[0]]*alphas[link[1]])
        return probs
    
    def construct_sample_network(probs):
        '''
        construct_sample_network(probs)

        Construct sample network according to the connection probability provided by probs.

        Parameters
        ----------
        probs: A dictionary contains the connection probability of the links(of G0, the complete network). Example: {(1, 3): 0.99, (1, 4): 0.96}.

        Returns
        -------
        Gsample: Constructed sample network according to probs in edgelist format.
        '''
        Gsample = []
        for i in probs:
            prob = probs[i]
            rand = random.random()
            if prob>=rand:
                Gsample.append(i)
        return Gsample

    def alphas_iteration(G0dict,G1dict,degrees,alphas_init:dict=None,iters:int=1000):
        """alphas_iteration iterate updating alphas for given iterations.

        Parameters
        ----------
        G0dict : _type_
            _description_
        G1dict : _type_
            _description_
        degrees : _type_
            _description_
        alphas_init : dict, optional
            _description_, by default None
        iters : int, optional
            _description_, by default 1000

        Returns
        -------
        _type_
            _description_
        """
        # initialize alphas
        alphas = {}.fromkeys(G1dict.keys(),1) if alphas_init == None else alphas_init
        for itering in range(iters):
            alphas_tem = {}.fromkeys(G1dict.keys(),1) # save the new alphas, update alphas until the end of iteration.
            for i in G1dict:
                Sigma = 0
                for j in G0dict[i]: # only the links exsit in G0 will be counted.
                    if j in G1dict.keys(): # if node j also in G1, otherwise degrees[j]=0
                        Sigma += 1 / (alphas[j] + 1 / alphas[i])
                alphas_tem[i] = Sigma / degrees[i]
            alphas.update(alphas_tem)
        return alphas 

    def cal_pos(a1elist:list,a2elist:list,alphas:dict):
        """cal_pos calculate the positive benchmark for a given alphas.

        Parameters
        ----------
        a1elist : list
            _description_
        a2elist : list
            _description_
        alphas : dict
            _description_

        Returns
        -------
        _type_
            _description_
        """
        # alphas of a1elist
        P1 = random_subnetwork.cal_probability(a2elist,a1elist,alphas=alphas) # only calculate the probability if a2elist
        Pv1 = {link: P1[link]*(1-P1[link]) for link in P1 }
        pos1_mean = sum([P1.get(link,0) for link in a2elist])
        pos1_sigma = np.sqrt(sum([Pv1.get(link,0) for link in a2elist]))   
        return pos1_mean,pos1_sigma

    def optimize_pos(a1elist, a2elist, searchSpace: list = None, iters_start=1000, pos_change_limit=1, iter_spacing=1000, max_iterations=20000):
            
        '''
        optimize_pos(a1elist,a2elist,iters_start=1000,pos_change_limit=1,iter_spacing=1000,max_iterations=20000)

        Optimize the pos by stopping iterating alphas at a given criterion. Return the pos by comparing randomized network1 to the original network2.

        Parameters
        ----------
        a1elist: network1 represented in the edgelist format.
        a2elist: network2 represented in the edgelist format.  
        searchSpace
        iters_start: the minimum iteration of alphas, default = 1000
        pos_change_limit: the stopping criterion based on the absolute change of pos between iter_spacing iterations.
        iter_spacing: check the pos change every iter_spacing.
        max_iterations: The maximum iterations.

        Returns
        -------
        cur_iter: the stopped iteration
        pos_mean: the pos mean
        pos_sigma: the pos sigma

        '''
        G1dict = formatting.edgelist_to_neighborhood(a1elist)
        a0elist = list(set(a1elist).union(a2elist))
        G0dict = formatting.edgelist_to_neighborhood(a0elist)

        G0dict = dict_remove_self(G0dict) # remove self-interactions
        G1dict = dict_remove_self(G1dict) # remove self-interactions
        degrees = random_subnetwork.cal_node_degree(G1dict) # reference degree sequence generated from G1
        # fisrt generate alphas for iters_start iterations
        alphas_prev = random_subnetwork.alphas_iteration(G0dict,G1dict,degrees,alphas_init=None,iters=iters_start)
        pos_mean_prev, pos_sigma_prev = random_subnetwork.cal_pos(a1elist,a2elist,alphas=alphas_prev)
        # check pos for every iter_spacing
        for i in range(iters_start+iter_spacing,max_iterations+iter_spacing,iter_spacing):
            alphas = random_subnetwork.alphas_iteration(G0dict,G1dict,degrees,alphas_init=alphas_prev,iters=iters_start)
            # check the change in pos
            pos_mean, pos_sigma = random_subnetwork.cal_pos(a1elist,a2elist,alphas=alphas)
            # check the absolute change in pos
            if abs(pos_mean - pos_mean_prev) < pos_change_limit:
                cur_iter = i # record the stopped iter
                break # stop iteration
            else:
                alphas_prev = alphas
                pos_mean_prev = pos_mean
                cur_iter = i

        return pos_mean, pos_sigma, cur_iter


class random_network:
    
    def optimize_alpha(G1dict0,iters=100):
        
        '''
        optimize_alpha(G1dict0,iters=100)

        Optimize the alpha for nodes in G1dict.

        Parameters
        ----------
        G1dict0: Reference network in neighborhood format that provides the node degree constriants.
        iters: The number of iterations for updating alphas.

        Returns
        -------
        alphas: A dictionary contains the optimized alphas for each node.
        '''
        G1dict = dict_remove_self(G1dict0) # remove self-interactions
        degrees = random_subnetwork.cal_node_degree(G1dict) # reference degree sequence generated from G1
        nodelist = G1dict.keys()
        alphas_tem = {}.fromkeys(nodelist,1)
        alphas = {}.fromkeys(nodelist,1)
        alphas_tem_value = np.array(dict_values(alphas_tem,nodelist))
        for itering in range(iters):
            Sigma = np.array([np.sum(ai/((ai*alphas_tem_value)+1)) for ai in alphas_tem_value]) - np.array([ai/(ai**2+1) for ai in alphas_tem_value])
            degree_value = np.array(dict_values(degrees,nodelist))
            alphas_tem_value = Sigma/degree_value
        for i,node in enumerate(nodelist):
            alphas[node] = alphas_tem_value[i]
        return alphas
    
    def cal_Pij(alphas,selfNodes):
        '''
        cal_Pij(alphas,selfNodes)

        Calculate the probability matrix for all possible combinations of nodes in alphas. Self-interactions Pii=1.

        Parameters
        ----------
        alphas: The optimized alphas for each node(of G1, the reference network).
        selfNodes: The nodes that have self-interactions in G1(the reference network).

        Returns
        -------
        Pij: The probability matrix following the order given by nodelist.
        nodelist: Provide the reference of the order of the probability matrix Pij.

        '''
        nodelist = sorted(alphas.keys())
        alphas_value = np.array([dict_values(alphas,nodelist)])
        Pij = 1/(1+np.dot(alphas_value.T,alphas_value))
        Pij = Pij - np.diag(np.diag(Pij))
        for selfNode in selfNodes:
            if selfNode in nodelist:
                index = nodelist.index(selfNode)
                Pij[index][index] = 1
        return Pij,nodelist

    def construct_random_network(Pij,nodelist,selfNodes):
        '''
        construct_random_network(Pij,nodelist), self-interactions are preserved.

        Construct the random network according to the given probability matrix Pij.

        Parameters
        ----------
        Pij: The probability matrix following the order given by nodelist.
        nodelist: Provide the reference of the order of the probability matrix Pij.
        selfNodes: The nodes that have self-interactions in G1(the reference network).

        Returns
        -------
        Gsample: random network in edgelist format.

        '''
        N = len(Pij)
        Rij = np.random.random(size=(N,N))
        Rij = np.triu(Rij)+np.triu(Rij,k=1).T
        Aij = Rij.copy()
        Aij[Pij<Rij]=0
        Aij[Pij>Rij]=1
        indices = np.where(np.triu(Aij)>0)
        nodelist = np.array(nodelist)
        Gsample = sort_elist(np.array([nodelist[indices[0]],nodelist[indices[1]]]).T)
        selfinters = [(node,node) for node in selfNodes]
        Gsample = list(set(Gsample).union(selfinters))
        return Gsample 


    def alphas_iteration(G1dict0: list, alphas_init: dict = None, iters: int = 100):
        """alphas_iteration iterate updating alphas for given iterations.

            Parameters
            ----------
            G0dict0 : _type_
                _description_
            alphas_init : dict, optional
                _description_, by default None
            iters : int, optional
                _description_, by default 1000

            Returns
            -------
            _type_
                _description_
            """

        G1dict = dict_remove_self(G1dict0)  # remove self-interactions
        nodelist = list(G1dict.keys())

        # reference degree sequence generated from G1alphas_tem = {}.fromkeys(nodelist,1)
        degrees = random_subnetwork.cal_node_degree(G1dict)
        degree_value = np.array(dict_values(degrees, nodelist))

        # initialize alphas
        alphas = {}.fromkeys(nodelist, 1) if alphas_init == None else alphas_init
        alphas_tem_value = np.array(dict_values(alphas, nodelist))

        for _ in range(iters):
            Sigma = np.array([np.sum(ai / ((ai * alphas_tem_value) + 1))
                            for ai in alphas_tem_value]) - np.array([ai / (ai**2 + 1) for ai in alphas_tem_value])
            alphas_tem_value = Sigma / degree_value

        for i, node in enumerate(nodelist):
            alphas[node] = alphas_tem_value[i]

        return alphas


    def cal_neg(a1elist: list, a2elist: list, alphas: dict):
        """cal_neg calculate the negative benchmark for a given alphas.

            Parameters
            ----------
            a1elist : list
                _description_
            a2elist : list
                _description_
            alphas : dict
                _description_

            Returns
            -------
            float,float
                neg1_mean, neg1_sigma
            """
        # only calculate the links in the comparing network(a2elist). Other links will never overlap with the comparing network.
        P1 = random_subnetwork.cal_probability(
            a2elist, a1elist, alphas)
        Pv1 = {link: P1[link]*(1-P1[link]) for link in P1}
        # one side
        neg1_mean = sum([P1.get(link, 0) for link in a2elist])
        neg1_sigma = np.sqrt(sum([Pv1.get(link, 0) for link in a2elist]))
        return neg1_mean, neg1_sigma


    def optimize_neg(a1elist, a2elist, iters_start=100, neg_change_limit=1, iter_spacing=100, max_iterations=2000):
        '''
            optimize_neg(a1elist,a2elist,iters_start=1000,pos_change_limit=1,iter_spacing=1000,max_iterations=20000)

            Optimize the neg by stopping iterating alphas at a given criterion.

            Parameters
            ----------
            a1elist: network1 represented in the edgelist format.
            a2elist: network2 represented in the edgelist format.
            iters_start: the minimum iteration of alphas, default = 100
            neg_change_limit: the stopping criterion based on the absolute change of neg between iter_spacing iterations.
            iter_spacing: check the neg change every iter_spacing.
            max_iterations: The maximum iterations.

            Returns
            -------
            cur_iter: the stopped iteration
            neg_mean: the neg mean
            neg_sigma: the neg sigma

            '''
        G1dict = formatting.edgelist_to_neighborhood(a1elist)
        a0elist = list(set(a1elist).union(a2elist))

        G1dict = dict_remove_self(G1dict)  # remove self-interactions
        # reference degree sequence generated from G1
        degrees = random_subnetwork.cal_node_degree(G1dict)

        # fisrt generate alphas for iters_start iterations
        alphas_prev = random_network.alphas_iteration(
            G1dict, alphas_init=None, iters=iters_start)
        neg_mean_prev, neg_sigma_prev = random_network.cal_neg(
            a1elist, a2elist, alphas=alphas_prev)
        # check neg for every iter_spacing
        for i in range(iters_start+iter_spacing, max_iterations+iter_spacing, iter_spacing):
                alphas = random_network.alphas_iteration(
                    G1dict, alphas_init=alphas_prev, iters=iters_start)
                # check the change in neg
                neg_mean, neg_sigma = random_network.cal_neg(
                    a1elist, a2elist, alphas=alphas)
                # check the absolute change in pos
                if abs(neg_mean - neg_mean_prev) < neg_change_limit:
                    cur_iter = i  # record the stopped iter
                    break  # stop iteration
                else:
                    alphas_prev = alphas
                    neg_mean_prev = neg_mean
                    cur_iter = i

        return neg_mean, neg_sigma, cur_iter
    
class helper:
    @staticmethod
    def get_uv(x, y, PPIr, uvJoin=True):
        '''
        get_uv(x, y, PPIr, uvJoin=True)

        Get all the uv paris that connect x,y by x-u-v-y. Interactions like x-u-x-y are excluded.
        
        Parameters
        ----------
        x: Node x in str format
        y: Node y in str format
        PPIr: Input PPI network in edge list formation. For example, N = {"A": {"B", "C"},"B":{"A"},"C":{"A"}}

        Returns
        -------
        uvPair, candidateUs, candidateVs
    
        '''
        candidateUs = PPIr[x]-{y} # exclude interactions x-u-x-y
        candidateVs = PPIr[y]-{x}
        if not uvJoin:
            candidateUs = candidateUs-candidateVs
            candidateVs = candidateVs-candidateUs
        uvPair = []
        for u in candidateUs:
            for v in candidateVs:
                if u not in PPIr[v]: continue
                uvPair.append([u,v])
        return uvPair, candidateUs, candidateVs
    
class methods:
    
    @staticmethod
    def L2(N,degree_normalized=True):
        '''
        L2(N,degree_normalized=True)

        Make predictions for all pairs based on L2 method. Interactions like x-y-y and input interactions are excluded.

        Parameters
        ----------
        N: Input PPI network in neighboorhood formation. For example, N = {"A": {"B", "C"},"B":{"A"},"C":{"A"}}
        degree_normalized: Denote whether to use L3 or L3DN.

        Returns
        -------
        scores, predictedPPIs

        '''
        scores, predictedPPIs = [], []
        allNodePairs = list(combinations(list(N.keys()), 2)) # tuple format
        for nodePair in allNodePairs:
            if nodePair[1] in N[nodePair[0]]: continue
            x, y = nodePair[0], nodePair[1]
            candidateUs = N[x]-{y} # exclude x-y-y
            score = 0
            for u in candidateUs:
                if len(N[u]) == 0: continue
                if y in N[u]:
                    score += 1/math.sqrt(len(N[u])) if degree_normalized==True else 1
            scores.append(score)
            predictedPPIs.append(nodePair)
        return scores, predictedPPIs
    
    @staticmethod    
    def L3(N,degree_normalized=True):
        '''
        L3(N,degree_normalized=True)

        Make predictions for all pairs based on L3 method. Interactions like x-u-x-y and input interactions are excluded.
        
        Parameters
        ----------
        N: Input PPI network in neighboorhood formation. For example, N = {"A": {"B", "C"},"B":{"A"},"C":{"A"}}
        degree_normalized: Denote whether to use L3 or L3DN.

        Returns
        -------
        scores, predictedPPIs
    
        '''
        scores, predictedPPIs = [], []
        allNodePairs = list(combinations(list(N.keys()), 2)) # tuple format
        for nodePair in allNodePairs:
            if nodePair[1] in N[nodePair[0]]: continue #对于已经在input中出现的PPI不再进行预测
            x, y = nodePair[0], nodePair[1]
            uvPair, _, _ = helper.get_uv(x, y, N)
            score = 0
            for [u, v] in uvPair:
                if math.sqrt(len(N[u])*len(N[v])) == 0: continue
                score += 1/math.sqrt(len(N[u])*len(N[v])) if degree_normalized==True else 1
            scores.append(score)
            predictedPPIs.append(nodePair)
        return scores, predictedPPIs
    
    class L3E:
        sim_f1 = lambda A, B: len(A&B)/len(A) # similarity function: simple ratio
        sim_f2 = lambda A, B: len(A&B)/len(A|B) # similarity function: jaccard index
        sim_f2Alt = lambda A, B: len(A&B) / (len(A|B)-1) if (len(A|B)-1) != 0 else 0
        sim_f1Alt = lambda A, B: len(A&B)/(len(A)-1) if (len(A)-1) != 0 else 0
        outer_index = lambda A, B: len(B)/len(A) # same for all scoringMethod because $B \in N(a)$

        @staticmethod
        def L3E(N, sim_name):
            if sim_name == 'f1': sim_f = methods.L3E.sim_f1
            elif sim_name == 'f2': sim_f = methods.L3E.sim_f2
            elif sim_name == 'f1Alt': sim_f = methods.L3E.sim_f1Alt
            elif sim_name == 'f2Alt': sim_f = methods.L3E.sim_f2Alt
            else: raise AttributeError("similarity function {} does not exist".format(sim_name))
            scores, predictedPPIs = [], []
            allNodePairs = list(combinations(list(N.keys()), 2))
            for nodePair in allNodePairs:
                if nodePair[1] in N[nodePair[0]]: continue
                x, y = nodePair[0], nodePair[1]
                uvPair, candidateUs, candidateVs = helper.get_uv(x, y, N, uvJoin=True)
                U, V = set([uv[0] for uv in uvPair]), set([uv[1] for uv in uvPair])
                score = 0
                for [u, v] in uvPair:
                    score += sim_f(N[v],N[x]) * sim_f(N[u],N[y]) * sim_f(N[u],V) * sim_f(N[v],U)
                score *= methods.L3E.outer_index(N[x],U) * methods.L3E.outer_index(N[y],V)
                scores.append(score)
                predictedPPIs.append(nodePair)
            return scores, predictedPPIs

