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
from bigg.extension.eval_.mmd import *
from bigg.extension.eval_.mmd_stats import *
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

def correct_lobster_topology_check(graphs):
  correct = 0
  true_lobsters = []
  for g in graphs:
      if is_lobster(g):
          correct += 1
          true_lobsters.append(g)
  return correct / len(graphs), true_lobsters

def correct_grid_topology_check(graphs):
    correct = 0
    true_grids = []
    for g in graphs:
        if is_grid(g):
            correct += 1
            true_grids.append(g)
    return correct / len(graphs), true_grids

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
            print("Not a lobster graph")
            print(graph.edges())
            return False
    return True  

def compute_probs(g, print_results = False):
    g_prime = nx.Graph(g.edges)
    leaves1 = [l for l in g_prime.nodes() if g_prime.degree(l) == 1]
    g_prime.remove_nodes_from(leaves1)
    leaves2 = [l for l in g_prime.nodes() if g_prime.degree(l) == 1]
    g_prime.remove_nodes_from(leaves2)
    
    backbone = sorted(list(g_prime.nodes))
    #non_backbone = list(reversed(sorted([n for n in g.nodes() if n not in backbone])))
    one_hop = []
    two_hop = []
    
    for n in leaves2:
        neighbors = list(g.neighbors(n))
        if n-1 in neighbors and n+1 in neighbors:
            backbone += [n]
        elif n-1 in neighbors and n-1 in backbone:
            backbone += [n]
        elif n+1 in neighbors and n+1 in backbone:
            backbone += [n]
        else:
            k = min(neighbors)
            if k in backbone:
                one_hop += [n]
            else:
                two_hop += [n]
    
    for n in leaves1:
        neighbors = list(g.neighbors(n))
        if n-1 in neighbors and n+1 in neighbors:
            backbone += [n]
        elif n-1 in neighbors and n-1 in backbone:
            backbone += [n]
        elif n+1 in neighbors and n+1 in backbone:
            backbone += [n]
        else:
            k = min(neighbors)
            if k in backbone:
                one_hop += [n]
            else:
                two_hop += [n]
    
    if print_results:
        print("backbone: ", backbone)
        print("one hop: ", one_hop)
        print("two hop: ", two_hop)
    
    if len(g) == len(two_hop):
        print(g.edges())
        p1_hat = 0.0
    
    else:
        p1_hat = len(one_hop) / (len(g) - len(two_hop))
    
    if p1_hat == 0.0:
        p2_hat = 0.0
    
    else:
        p2_hat = len(two_hop) / (len(one_hop) + len(two_hop))
    
    if print_results:
        print("p1_hat: ", p1_hat)
        print("p2_hat: ", p2_hat)
    
    return p1_hat, p2_hat



def estimate_p(graphs):
    p1s = []
    p2s = []
    
    for g in graphs:
        p1_hat, p2_hat = compute_probs(g)
        p1s += [p1_hat]
        p2s += [p2_hat]
    
    p1  = sum(p1s) / len(p1s)
    p2 = sum(p2s) / len(p2s)
    
    print("P1 est: ", p1)
    print("P2 est: ", p2)
    return p1, p2



def is_grid(graph):
    res = True
    g = nx.Graph(graph.edges())
    
    bad_nodes = [x for x in g.nodes() if g.degree(x) not in range(2,5)]
    if len(bad_nodes) > 0:
        return False
    
    corners = [x for x in g.nodes() if g.degree(x) == 2]
    if len(corners) != 4:
        return False
    
    p_lens = [len(nx.shortest_path(g, corners[0], x)) for x in corners[1:]]
    m = min(p_lens)
    n = max(p_lens) - m + 1
    
    if m * n != len(g):
        return False
    
    sides = [x for x in g.nodes() if g.degree(x) == 3]
    interior = [x for x in g.nodes() if g.degree(x) == 4]
    
    if len(sides) != 2*(m + n) - 8 or len(interior) != m * n - 2*(m + n) + 4:
        return False
    return True

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
  

  #print("mu-hat, mu_lo, mu_up, s, s_lo, s_up, mean_tree_var, tree_var_lo, tree_var_up")
  results = [xbar, mu_lo, mu_up, s, s_lo, s_up, mean_tree_var**0.5, tree_var_lo**0.5, tree_var_up**0.5]
  results_rounded = np.round(results, 3)
  print("Num of trees")
  print(len(graphs))
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
  return results
  
def get_graph_stats(out_graphs, test_graphs, graph_type):
    if graph_type == "tree":
        prop, true_trees = correct_tree_topology_check(out_graphs)
        print("Proportion Correct Topology: ", prop)
        true_trees_edges = []
        true_train_edges = []
        
        #### TESTING MMD
        test = degree_stats(out_graphs, test_graphs)
        print("MMD Test on Degree Stats: ", test)
        test2 = spectral_stats(out_graphs, test_graphs, False)
        print("MMD on Specta of L Normalized, Unweighted: ", test2)
        test3 = spectral_stats(out_graphs, test_graphs, True)
        print("MMD on Specta of L Normalized, Weighted: ", test3)
        test4 = mmd_weights_only(out_graphs, test_graphs, gaussian_emd)
        print("MMD on Weights Only: ", test4)
        #test3 = clustering_stats(out_graphs, test_graphs)
        #print("MMD on Clustering Coefficient: ", test3)
        
        #if len(true_trees) > 10000:
        #    for tree in true_trees:
        #        e = tree.edges()
        #        true_trees_edges.append(e)
        #    for tree in ordered_train_graphs:
        #        tree_graph = graph_from_adj(tree)
        #        e = tree_graph.edges()
        #        if e not in true_train_edges:
        #            true_train_edges.append(e)
        #    in_train = 0
        #    test_check = 0
        #    for tree_edges in true_trees_edges:
        #        if tree_edges in true_train_edges:
        #            in_train += 1
        #        else:
        #            print("Tree not in training set")
        #            print(tree_edges)
        #        if tree_edges in true_trees_edges:
        #            test_check += 1
        #    print("Trees in training: ", in_train)
        #    print("New trees: ", len(true_trees_edges) - in_train)
        #    print(true_train_edges)
        #    print("Test Check: ", test_check)
        if True: #cmd_args.weighted:
            #test_stats = tree_weight_statistics(true_trees)
            print("Weight stats of ALL graphs")
            test_stats2 = tree_weight_statistics(out_graphs)
    
    elif graph_type == "lobster":
        prop, true_lobsters = correct_lobster_topology_check(out_graphs)
        print("Proportion Correct Lobster Graphs: ", prop)
        xbars = []
        vars_ = []
        num_nodes = []
        num_edges = []
        for lobster in out_graphs:
            weights = []
            num_nodes.append(len(lobster))
            num_edges.append(len(lobster.edges()))
            if False:
                for (n1, n2, w) in lobster.edges(data=True):
                    w = w['weight'] #np.log(np.exp(w['weight']) - 1)
                    weights.append(w)
                xbars.append(np.mean(weights))
                vars_.append(np.var(weights, ddof = 1))
        
        if False: #cmd_args.weighted:
            mu_lo = np.mean(xbars) - 1.96 * np.std(xbars) / len(xbars)**0.5
            mu_up = np.mean(xbars) + 1.96 * np.std(xbars) / len(xbars)**0.5
            
            var_lo = np.mean(vars_) - 1.96 * np.std(vars_) / len(vars_)**0.5
            var_up = np.mean(vars_) + 1.96 * np.std(vars_) / len(vars_)**0.5
            
            print("Mean Estimates: ", np.mean(xbars), (mu_lo, mu_up))
            print("Var Estimates: ", np.mean(vars_)**0.5, (var_lo**0.5, var_up**0.5))
            
        print("Num Nodes: ", np.mean(num_nodes), (min(num_nodes), max(num_nodes)))
        print("Num Edges: ", np.mean(num_edges), (min(num_edges), max(num_edges)))
        
        #p1, p2 = estimate_p(true_lobsters)
        test = degree_stats(out_graphs, test_graphs)
        print("MMD Test on Degree Stats: ", test)
        test2 = spectral_stats(out_graphs, test_graphs, False)
        print("MMD on Specta of L Normalized, Unweighted: ", test2)
        test3 = spectral_stats(out_graphs, test_graphs, True)
        print("MMD on Specta of L Normalized, Weighted: ", test3)
        if False:
            test4 = mmd_weights_only(out_graphs, test_graphs, gaussian_emd)
            print("MMD on Weights Only: ", test4)
        #test3 = clustering_stats(out_graphs, test_graphs)
        #print("MMD on Clustering Coefficient: ", test3)
    
    elif graph_type == "grid":
        prop, true_lobsters = correct_grid_topology_check(out_graphs)
        print("Proportion Correct Topology: ", prop)
    
    elif graph_type == "db":
        test = degree_stats(out_graphs, test_graphs)
        print("MMD Test on Degree Stats: ", test)
        test2 = spectral_stats(out_graphs, test_graphs, False)
        print("MMD on Specta of L Normalized, Unweighted: ", test2)
        test3 = spectral_stats(out_graphs, test_graphs, True)
        print("MMD on Specta of L Normalized, Weighted: ", test3)
        test4 = mmd_weights_only(out_graphs, test_graphs, gaussian_emd)
        print("MMD on Weights Only: ", test4)
        test5 = clustering_stats(out_graphs, test_graphs)
        print("MMD on Clustering Coefficient: ", test5)
        #test6 = motif_stats(out_graphs, test_graphs)
        #print("MMD on Orbit: ", test6)    
    else:
        print("Graph Type not yet implemented")
    return 0
    
    

    
    













