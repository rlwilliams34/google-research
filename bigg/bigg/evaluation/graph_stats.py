from scipy.stats.distributions import chi2
import networkx as nx
import numpy as np
import torch
import random
#from numpy import random
from torch import nn
from torch.nn.parameter import Parameter
import pandas as pd
import os
import scipy
from bigg.evaluation.mmd import *
from bigg.evaluation.mmd_stats import *
from bigg.common.configs import cmd_args, set_device

		## Topology Check Functions
def correct_tree_topology_check(graphs):
  correct = 0
  true_trees = []
  for g in graphs:
    if is_bifurcating_tree(g):
        correct += 1
        true_trees.append(g)
  return correct / len(graphs), true_trees

def correct_tree_topology_check_two(graphs):
    props = []
    
    for g in graphs:
        root = [0]
        leaves = [n for n in g.nodes() if g.degree(n) == 1]
        internal = [n for n in g.nodes() if g.degree(n) == 3]
        good_nodes = root + leaves + internal
        props.append(len(good_nodes) / len(g))
    
    avg_prop = np.mean(props)
    return avg_prop

def correct_lobster_topology_check(graphs):
  correct = 0
  true_lobsters = []
  for g in graphs:
      if is_lobster(g):
          correct += 1
          true_lobsters.append(g)
  return correct / len(graphs), true_lobsters

# def correct_grid_topology_check(graphs):
#     correct = 0
#     true_grids = []
#     for g in graphs:
#         if is_grid(g):
#             correct += 1
#             true_grids.append(g)
#     return correct / len(graphs), true_grids

def is_bifurcating_tree(g):
    if nx.is_tree(g):
        leaves = [n for n in g.nodes() if g.degree(n) == 1]
        internal = [n for n in g.nodes() if g.degree(n) == 3]
        root = [n for n in g.nodes() if g.degree(n) == 2]
        if 2*len(leaves) - 1 == len(g) and len(leaves) == len(internal) + 2 and len(root) == 1 and len(leaves) + len(internal)+ len(root) == len(g):
            return True
    return False

def is_lobster(graph):
    if not nx.is_tree(graph):
        return False
    g = nx.Graph(graph.edges())
    leaves = [l for l in g.nodes() if g.degree(l) == 1]
    g.remove_nodes_from(leaves)
    big_n = [n for n in g.nodes() if g.degree(n) >= 3]
    
    for n in big_n:
        big_neighbors = [x for x in g.neighbors(n) if g.degree(x) >= 2]
        if len(big_neighbors) > 2:
     	    return False
    return True

# def compute_probs(g, print_results = False):
#     g_prime = nx.Graph(g.edges)
#     leaves1 = [l for l in g_prime.nodes() if g_prime.degree(l) == 1]
#     g_prime.remove_nodes_from(leaves1)
#     leaves2 = [l for l in g_prime.nodes() if g_prime.degree(l) == 1]
#     g_prime.remove_nodes_from(leaves2)
#     
#     backbone = sorted(list(g_prime.nodes))
#     #non_backbone = list(reversed(sorted([n for n in g.nodes() if n not in backbone])))
#     one_hop = []
#     two_hop = []
#     
#     for n in leaves2:
#         neighbors = list(g.neighbors(n))
#         if n-1 in neighbors and n+1 in neighbors:
#             backbone += [n]
#         elif n-1 in neighbors and n-1 in backbone:
#             backbone += [n]
#         elif n+1 in neighbors and n+1 in backbone:
#             backbone += [n]
#         else:
#             k = min(neighbors)
#             if k in backbone:
#                 one_hop += [n]
#             else:
#                 two_hop += [n]
#     
#     for n in leaves1:
#         neighbors = list(g.neighbors(n))
#         if n-1 in neighbors and n+1 in neighbors:
#             backbone += [n]
#         elif n-1 in neighbors and n-1 in backbone:
#             backbone += [n]
#         elif n+1 in neighbors and n+1 in backbone:
#             backbone += [n]
#         else:
#             k = min(neighbors)
#             if k in backbone:
#                 one_hop += [n]
#             else:
#                 two_hop += [n]
#     
#     if print_results:
#         print("backbone: ", backbone)
#         print("one hop: ", one_hop)
#         print("two hop: ", two_hop)
#     
#     if len(g) == len(two_hop):
#         print(g.edges())
#         p1_hat = 0.0
#     
#     else:
#         p1_hat = len(one_hop) / (len(g) - len(two_hop))
#     
#     if p1_hat == 0.0:
#         p2_hat = 0.0
#     
#     else:
#         p2_hat = len(two_hop) / (len(one_hop) + len(two_hop))
#     
#     if print_results:
#         print("p1_hat: ", p1_hat)
#         print("p2_hat: ", p2_hat)
#     
#     return p1_hat, p2_hat

# def estimate_p(graphs):
#     p1s = []
#     p2s = []
#     
#     for g in graphs:
#         p1_hat, p2_hat = compute_probs(g)
#         p1s += [p1_hat]
#         p2s += [p2_hat]
#     
#     p1  = sum(p1s) / len(p1s)
#     p2 = sum(p2s) / len(p2s)
#     
#     print("P1 est: ", p1)
#     print("P2 est: ", p2)
#     return p1, p2
# 
# def is_grid(graph):
#     res = True
#     g = nx.Graph(graph.edges())
#     
#     bad_nodes = [x for x in g.nodes() if g.degree(x) not in range(2,5)]
#     if len(bad_nodes) > 0:
#         return False
#     
#     corners = [x for x in g.nodes() if g.degree(x) == 2]
#     if len(corners) != 4:
#         return False
#     
#     p_lens = [len(nx.shortest_path(g, corners[0], x)) for x in corners[1:]]
#     m = min(p_lens)
#     n = max(p_lens) - m + 1
#     
#     if m * n != len(g):
#         return False
#     
#     sides = [x for x in g.nodes() if g.degree(x) == 3]
#     interior = [x for x in g.nodes() if g.degree(x) == 4]
#     
#     if len(sides) != 2*(m + n) - 8 or len(interior) != m * n - 2*(m + n) + 4:
#         return False
#     return True

def group_lobster_edges(g):
    backbone, one_hop, two_hop = group_lobster_nodes(g)
    
    edge_1, edge_2, edge_3 = [], [], []
    #print("NEW GRAPH")
    #print(g.edges())
    
    for (n1, n2, w) in g.edges(data=True):
        w = w['weight']
        if n1 in backbone and n2 in backbone:
            #print("Backbone edge: ")
            #print(n1, n2, w)
            edge_1.append(w)
        
        elif n1 in two_hop or n2 in two_hop:
            edge_3.append(w)
        
        else:
            edge_2.append(w)
    
    return edge_1, edge_2, edge_3

def group_lobster_nodes(g):
    g_prime = nx.Graph(g.edges)
    leaves1 = [l for l in g_prime.nodes() if g_prime.degree(l) == 1]
    g_prime.remove_nodes_from(leaves1)
    
    if len(g_prime.edges()) == 1:
        leaves2 = []
        
    else:
        leaves2 = [l for l in g_prime.nodes() if g_prime.degree(l) == 1]
        g_prime.remove_nodes_from(leaves2)
    
    backbone = sorted(list(g_prime.nodes))
    one_hop = []
    two_hop = []
    
    for n in leaves2:
        neighbors = list(g.neighbors(n))
        k = min(neighbors)
        if k in backbone:
            one_hop += [n]
        else:
            two_hop += [n]
    
    for n in leaves1:
        neighbors = list(g.neighbors(n))
        k = min(neighbors)
        if k in backbone:
            one_hop += [n]
        else:
            two_hop += [n]
    
    return backbone, one_hop, two_hop


# def lobster_weight_statistics(graphs):
#     means_1, means_2, means_3 = [], [], []
#     
#     for g in graphs:
#         edge_1, edge_2, edge_3 = group_lobster_edges(g)
#         
#         if len(edge_1) > 0:
#             means_1.append(np.mean(edge_1))
#         
#         if len(edge_2) > 0:
#             means_2.append(np.mean(edge_2))
#         
#         if len(edge_3) > 0:
#             means_3.append(np.mean(edge_3))
#     
#     mu_1_lo = np.percentile(means_1, 2.5)
#     mu_1_hi = np.percentile(means_1, 97.5)
#     print("Mean 1 Estimate", np.mean(means_1))
#     print('Empirical Interval: ', ' (' + str(mu_1_lo) + ',' + str(mu_1_hi) + ')')
#     
#     mu_2_lo = np.percentile(means_2, 2.5)
#     mu_2_hi = np.percentile(means_2, 97.5)
#     print("Mean 2 Estimate", np.mean(means_2))
#     print('Empirical Interval: ', ' (' + str(mu_2_lo) + ',' + str(mu_2_hi) + ')')
#     
#     mu_3_lo = np.percentile(means_3, 2.5)
#     mu_3_hi = np.percentile(means_3, 97.5)
#     print("Mean 3 Estimate", np.mean(means_3))
#     print('Empirical Interval: ', ' (' + str(mu_3_lo) + ',' + str(mu_3_hi) + ')')

def lobster_weight_statistics(graphs):
    means = []
    vars_ = []
    
    for g in graphs:
        weights = []
        for (n1, n2) in g.edges():
            weights.append(g[n1][n2]['weight'])
        
        means.append(np.mean(weights))
        vars_.append(np.var(weights, ddof = 1))
        
    mu_lo = np.percentile(means, 2.5)
    mu_hi = np.percentile(means, 97.5)
    print("Mean Estimate", np.mean(means))
    print('Empirical Interval: ', ' (' + str(mu_lo) + ',' + str(mu_hi) + ')')   
    
    var_lo = np.percentile(vars_, 2.5)
    var_hi = np.percentile(vars_, 97.5)
    print("Var Estimate", np.mean(vars_))
    print('Empirical Interval: ', ' (' + str(var_lo) + ',' + str(var_hi) + ')')    
    
            

def tree_weight_statistics(graphs, transform = False):
  ## Returns summary statistics on weights for graphs
  weights = []
  tree_vars = []
  tree_means = []

  for T in graphs:
    T_weights = []
    for (n1, n2, w) in T.edges(data = True):
      if transform:
        t = np.log(np.exp(w['weight']) - 1)
      else:
        t = w['weight']
      T_weights.append(t)
      weights.append(t)
    tree_vars.append(np.var(T_weights, ddof = 1))
    tree_means.append(np.mean(T_weights))

  xbar = np.mean(weights)
  s = np.std(weights, ddof = 1)
  n = len(weights)

  mu_lo = np.round(xbar - 1.96 * s / n**0.5, 3)
  mu_up = np.round(xbar + 1.96 * s / n**0.5, 3)

  s_lo = np.round(s * (n-1)**0.5 * (1/chi2.ppf(0.975, df = n-1))**0.5, 3)
  s_up = np.round(s * (n-1)**0.5 * (1/chi2.ppf(0.025, df = n-1))**0.5, 3)

  mean_tree_var = np.mean(tree_vars)
  tree_var_lo = mean_tree_var - 1.96 * np.std(tree_vars, ddof = 1) / len(tree_vars)**0.5
  tree_var_up = mean_tree_var + 1.96 * np.std(tree_vars, ddof = 1) / len(tree_vars)**0.5
  
  mu_t_lo = np.percentile(tree_means, 2.5)
  mu_t_hi = np.percentile(tree_means, 97.5)
  
  s_t_lo = np.percentile(tree_vars, 2.5)**0.5
  s_t_hi = np.percentile(tree_vars, 97.5)**0.5
  
  print("Num of trees")
  print(len(graphs))
  
  results = [xbar, mu_lo, mu_up, s, s_lo, s_up, mean_tree_var**0.5, tree_var_lo**0.5, tree_var_up**0.5]
  results_rounded = np.round(results, 3)
  
  print("Mean Estimates")
  print(results_rounded[0])
  print('95% CI: ', ' (' + str(results_rounded[1]) + ',' + str(results_rounded[2]), ')')
  print('Empirical Interval: ', ' (' + str(mu_t_lo) + ',' + str(mu_t_hi) + ')')
  
  print("SD Estimates")
  print(results_rounded[3])
  print('95% CI: ', ' (' + str(results_rounded[4]) + ',' + str(results_rounded[5]), ')')  
  
  print("Within Tree Variability")
  print(results_rounded[6])
  print('95% CI: ', ' (' + str(results_rounded[7]) + ',' + str(results_rounded[8]), ')')
  print('Empirical Interval: ', ' (' + str(s_t_lo) + ',' + str(s_t_hi) + ')')

def get_mmd_stats(out_graphs, test_graphs):
    mmd_degree = degree_stats(out_graphs, test_graphs)
    print("MMD Test on Degree Stats: ", mmd_degree)
    
    mmd_spectral_unweighted = spectral_stats(out_graphs, test_graphs, False)
    print("MMD on Specta of L Normalized, Unweighted: ", mmd_spectral_unweighted)
    
    mmd_cluster = clustering_stats(out_graphs, test_graphs)
    print("MMD on Clustering Coefficient: ", mmd_cluster)
    
    if cmd_args.has_edge_feats:
        mmd_sepctral_weighted = spectral_stats(out_graphs, test_graphs, True)
        print("MMD on Specta of L Normalized, Weighted: ", mmd_sepctral_weighted)
        
        mmd_weights = mmd_weights_only(out_graphs, test_graphs, gaussian_tv)
        print("MMD on Weights Only: ", mmd_weights)
    
    #mmd_orbit = motif_stats(out_graphs, test_graphs)
    mmd_orbit = orbit_stats_all(out_graphs, test_graphs)
    print("MMD on Orbit: ", mmd_orbit)


def get_graph_stats(out_graphs, test_graphs, graph_type):
    if graph_type == "tree":
        prop, _ = correct_tree_topology_check(out_graphs)
        print("Proportion Correct Topology: ", prop)
        
        prop2 = correct_tree_topology_check_two(out_graphs)
        print("Alt Proportion Correct Topology: ", prop2)
        
        if cmd_args.has_edge_feats:
            print("Weight stats of ALL graphs")
            test_stats2 = tree_weight_statistics(out_graphs)
        
        if test_graphs is None:
            return prop
                
        get_mmd_stats(out_graphs, test_graphs)
        
    
    elif graph_type == "lobster":
        prop, true_lobs = correct_lobster_topology_check(out_graphs)
        print("Proportion Correct Lobster Graphs: ", prop)
        
        num_nodes = []
        num_edges = []
        for lobster in out_graphs:
            num_nodes.append(len(lobster))
            num_edges.append(len(lobster.edges()))
        print("Num Nodes: ", np.mean(num_nodes), (min(num_nodes), max(num_nodes)))
        print("Num Edges: ", np.mean(num_edges), (min(num_edges), max(num_edges)))
        
        if cmd_args.has_edge_feats:
            lobster_weight_statistics(out_graphs)
            print("checking with true lobsters")
            lobster_weight_statistics(true_lobs)
        
        if test_graphs is None:
            return prop
        
        get_mmd_stats(out_graphs, test_graphs)
    
#     elif graph_type == "grid":
#         prop, true_lobsters = correct_grid_topology_check(out_graphs)
#         print("Proportion Correct Topology: ", prop)
    
    elif graph_type == "db":
        weights = []
        for g in out_graphs:
            for (n1, n2, w) in g.edges(data=True):
                weights.append(w['weight']) 
        print("Mean weight: ", np.mean(weights))
        print("SD Weight: ", np.std(weights, ddof = 1))
        
        num_nodes = []
        num_edges = []
        for lobster in out_graphs:
            num_nodes.append(len(lobster))
            num_edges.append(len(lobster.edges()))
        print("Num Nodes: ", np.mean(num_nodes), (min(num_nodes), max(num_nodes)))
        print("Num Edges: ", np.mean(num_edges), (min(num_edges), max(num_edges)))
        
        get_mmd_stats(out_graphs, test_graphs)
    
    elif graph_type == "er":
        probs = []
        weights = []
        for g in out_graphs:
            n = len(g)
            if n <= 1:
                continue
            m = len(g.edges())
            p = 2 * m / (n * (n - 1))
            probs.append(p)
            for (n1, n2, w) in g.edges(data=True):
                w_sm = np.log(np.exp(w['weight']) - 1)
                weights.append(w_sm)
        
        num_nodes = []
        num_edges = []
        for lobster in out_graphs:
            num_nodes.append(len(lobster))
            num_edges.append(len(lobster.edges()))
        print("Num Nodes: ", np.mean(num_nodes), (min(num_nodes), max(num_nodes)))
        print("Num Edges: ", np.mean(num_edges), (min(num_edges), max(num_edges)))
        
        p_lo = np.percentile(probs, 2.5)
        p_hi = np.percentile(probs, 97.5)
        p_mu = np.mean(probs)
        print("Mean prob of edge existence: ", p_mu)
        print("95% Credible Interval: ", "(", p_lo, ", ", p_hi, ")")
        
        print("Mean SM weight: ", np.mean(weights))
        print("SD SM Weight: ", np.std(weights, ddof = 1))
        
        weights = []
        for g in test_graphs:
            for (n1, n2, w) in g.edges(data=True):
                weights.append(w['weight'])
        
        print("Mean Test weight: ", np.mean(weights))
        print("SD Test Weight: ", np.std(weights, ddof = 1))
        
        get_mmd_stats(out_graphs, test_graphs)
    
    else:
        print("Graph Type not yet implemented")
    return 0
    













