import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import math
from sympy import Matrix


T=0.1 #time step

class Node:
    node_num=0
    def __init__(self,index,state):
        self.index=index
        self.state=state
        self.neighbor=[]
        self.neighbor_info=[]
        Node.node_num+=1

    def get_info(self,A_matrix,state_list):
        state_neighbor=0
        for i in range (Node.node_num):
            if(A_matrix[i][self.index]>0):
                state_neighbor += (state_list[i] - self.state)
        self.state=self.state+state_neighbor*T

    def reconstruction(self,A_matrix,node_1,edge_list):
        node_0_list=get_chlid(A_matrix,node_1)
        if(len(node_0_list)>0):
            #检查 node_1和node_0是否连通
            for node in node_0_list:
                if (A_matrix[node][self.index] > 0):
                    return
            degree=get_degree(A_matrix)
            node_1_=node_0_list[np.argmax([degree[i] for i in node_0_list])]
            edge_list.append((node_1_,self.index))
            A_matrix[node_1_][self.index] = 1
        else:
            node_2_list=get_parent(A_matrix,node_1)
            for node in node_2_list:
                if (A_matrix[node][self.index]==0):
                    edge_list.append((node,self.index))
                    A_matrix[node][self.index]=1

def attack_node(G,A_matrix,time):

    A_matrix_ = np.array(A_matrix, copy=True)
    #选取节点进行移除
    node_1=node_set[random.randint(0,len(node_set)-1)]
    node_set.remove(node_1)
    G.remove_node(node_1)
    for i in range(A_matrix.shape[0]):
        A_matrix[i][node_1]=0
        A_matrix[node_1][i]=0

    remove_edge_list=[0 for i in range(node_num)]
    edge_list = []
    node_2_list = []
    node_0_list = []
    for j in range(A_matrix_.shape[0]):
        if (A_matrix_[node_1][j] > 0):
            node_0_list.append(j)
            edge_list.append((node_1, j))
            remove_edge_list[j]+=1
        if (A_matrix_[j][node_1] > 0):
            node_2_list.append(j)
            edge_list.append((j, node_1))
            remove_edge_list[j] += 1
    # print("攻击节点:{},node_2:{},node_0:{}".format(node_1, node_2_list,node_0_list))
    # 绘制攻击图像
    plt.subplot(row, col,  plot_index[time])
    plt.title('Node %s is attacked'%node_1,font1, y=-0.1)
    nx.draw(G, pos=pos,with_labels=is_label,node_color=color[0],edge_color='lightgray')
    nx.draw_networkx_nodes(G, pos=pos, nodelist=node_2_list, node_color=color[1])
    nx.draw_networkx_nodes(G, pos=pos, nodelist=node_0_list, node_color=color[2])
    mutual_node = list(set(node_2_list) & set(node_0_list))
    nx.draw_networkx_nodes(G, pos=pos, nodelist=[node_1], node_color='red')
    nx.draw_networkx_nodes(G, pos=pos, nodelist=mutual_node, node_color=color[3])
    nx.draw_networkx_edges(G, pos=pos, edgelist=edge_list, edge_color='r', style='dashed')
    return A_matrix,node_2_list,node_0_list,node_1,remove_edge_list

def recover_edge(G,A_matrix,node_2_list,node_0_list,node_1,time):

    plt.subplot(row,col, plot_index[time])
    plt.title('Recover from attacked node %s'%node_1,font1, y=-0.1)
    nx.draw(G, pos=pos,with_labels=is_label,node_color=color[0],edge_color='lightgray')
    nx.draw_networkx_nodes(G, pos=pos, nodelist=node_2_list, node_color=color[1])
    nx.draw_networkx_nodes(G, pos=pos, nodelist=node_0_list, node_color=color[2])
    mutual_node = list(set(node_2_list) & set(node_0_list))
    nx.draw_networkx_nodes(G, pos=pos, nodelist=mutual_node, node_color=color[3])

    #获取node_0_list的权重
    recover_edge_list=[0 for i in range(node_num)]
    node_0_list_weight=[]
    for node_0 in node_0_list:
        weight=[0,0]
        node_0_parent_list = get_parent(A_matrix, node_0)
        node_0_grandparent_list=get_grandparent(A_matrix,node_0)
        node_0_child_list=get_chlid(A_matrix,node_0)
        weight[0]=len((set(node_0_parent_list)|set(node_0_grandparent_list))&set(node_2_list))+len(set(node_0_child_list)&set(node_0_list))
        weight[1]=len((set(node_0_parent_list)|set(node_0_grandparent_list))&set(node_0_list))
        node_0_list_weight.append(weight)

    weight_list=np.array(node_0_list_weight)
    weght_1_min=min(weight_list[:,1])
    temp=-1
    index=-1
    for i in range(len(node_0_list_weight)):
        if(node_0_list_weight[i][1]==weght_1_min):
            if(node_0_list_weight[i][0]>temp):
                index=i
    #根据权重，选出目标节点
    node_0_target=node_0_list[index]
    # print("weight:{}".format(node_0_list_weight))
    # print("target_node_0:{},".format(node_0_target))
    edge_list = []
    for node_0 in node_0_list:
        if (node_0 != node_0_target):
            node_0_parent_list = get_parent(A_matrix, node_0)
            node_0_grandparent_list = get_grandparent(A_matrix, node_0)
            # print(set(node_0_parent_list) | set(node_0_grandparent_list))
            if (node_0_target not in (set(node_0_parent_list) | set(node_0_grandparent_list))):
                edge_list.append((node_0_target, node_0))
                A_matrix[node_0_target][node_0] = 1
                recover_edge_list[node_0]+=1
    for node_2 in node_2_list:
        if (node_2 != node_0_target):
            node_0_parent_list = get_parent(A_matrix, node_0_target)
            node_0_grandparent_list=get_grandparent(A_matrix,node_0_target)
            # print(set(node_0_parent_list) | set(node_0_grandparent_list))
            if(node_2 not in (set(node_0_parent_list) | set(node_0_grandparent_list))):
                edge_list.append((node_2, node_0_target))
                A_matrix[node_2][node_0_target] = 1
                recover_edge_list[node_0_target]+=1

    # print("edge_list:{}".format(edge_list))

    if(edge_list):
        G.add_edges_from(edge_list)
        nx.draw_networkx_edges(G, pos=pos, edgelist=edge_list, edge_color=color[4])
    return A_matrix,recover_edge_list

def generate_random_graph(n,m):
    node_list = []
    graph = {}
    for node in range(n):  # 循环创建结点
        node_list.append(node)

    for node in node_list:
        graph[node] = []  # graph为dict结构，初始化为空列表

    #choose root
    unassigned_node=[i for i in range(n)]
    assigned_node=[]
    root= random.randint(0, n - 1)
    assigned_node.append(root)
    unassigned_node.remove(root)
    for node in node_list:  # 创建有向生成树
        if(node in set(unassigned_node)):
            index=random.randint(0,len(assigned_node)-1)
            # graph[node].append(assigned_node[index])
            graph[assigned_node[index]].append(node)
            assigned_node.append(node)
            unassigned_node.remove(node)
    #连接root
    unassigned_node = [i for i in range(n)]
    assigned_node = []
    assigned_node.append(root)
    unassigned_node.remove(root)
    for node in node_list:
        if (node in set(unassigned_node)):
            index = random.randint(0, len(assigned_node) - 1)
            graph[node].append(assigned_node[index])
            assigned_node.append(node)
            unassigned_node.remove(node)

    add_edge_num=m-2*(n-1)
    count=0
    while(True):
        node_set = [i for i in range(n)]
        index_a = random.randint(0, n - 1)
        node_set.remove(index_a)
        index_ = random.randint(0, n - 2)
        index_b=node_set[index_]
        if index_b not in graph[index_a]:
            graph[index_a].append(index_b)
            # graph[index_b].append(index_a)
            count+=1
        if(count>=add_edge_num):
            break
    return graph

def get_parent(A_matrix,index):
    parent_node=[]
    for i in range(A_matrix.shape[0]):
        if(A_matrix[i][index]>0):
            parent_node.append(i)
    # in_edge=[i for i in G.in_edges(index)]
    # parent_node=[in_edge[j][0] for j in range(len(in_edge))]
    return parent_node

def get_grandparent(A_matrix,index):
    grandparent_node=[]
    parent_node=get_parent(A_matrix,index)
    for parent in parent_node:
        for i in range(node_num):
            if (A_matrix[i][parent] > 0):
                grandparent_node.append(i)
    return grandparent_node

def get_chlid(A_matrix,index):
    child_node=[]
    for i in range(A_matrix.shape[0]):
        if(A_matrix[index][i]>0):
            child_node.append(i)
    # out_edge=[i for i in G.out_edges(0)]
    # child_node=[out_edge[j][0] for j in range(len(out_edge))]
    return child_node

def get_outdegree(A_matrix):
    return np.sum(A_matrix,axis=1)

def get_indegree(A_matrix):
    return np.sum(A_matrix,axis=0)

def get_degree(A_matrix):
    return np.sum(A_matrix,axis=0)+np.sum(A_matrix,axis=1)


if __name__ == '__main__':
    # Drawing parameters
    color=['#95D1CC','#F56D91','yellow','orange','green']
    is_label = True
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 20, }
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 16, }

    plt.rcParams['font.serif'] = ['Times New Roman']

    # Topology parameters
    node_num =20
    tau = 1.4
    edge_num = int(node_num * tau)

    # Draw reconfiguration process
    plt.figure(figsize=(12, 12))
    iter_num=80
    row = 3
    col = 3
    plot_index=[2,3,4,5,6,7,8,9]
    plt.subplot(row, col, 1)
    plt.title('The original graph',font1,y=-0.1)
    graph = generate_random_graph(node_num, edge_num)
    G = nx.DiGraph(graph)
    pos = nx.kamada_kawai_layout(G)
    nx.draw(G, pos=pos,with_labels=is_label,node_color=color[0],edge_color='lightgray')
    node_set = [i for i in range(node_num)]
    A_matrix = nx.adj_matrix(G).toarray()

    # State value initialization
    state_list=[]
    avg=10
    for i in range(0, math.ceil(node_num / 2)):
        state_list.append(random.randint(0, 20))
    for i in range(node_num - math.ceil(node_num / 2)):
        state_list.append(2 * avg - state_list[i])
    # Generate nodes
    node_list=[]
    for i in range(node_num):
        node=Node(i,state_list[i])
        node_list.append(node)

    control_input_list = []
    state_recorder = []
    attack_index = []
    time = 0
    for i in range(iter_num):
        state_recorder.append(state_list)
        for node in node_list:
            node.get_info(A_matrix, state_list)
        if (i>0 and i %10==0 and time<7):
            # Input：G，A_matrix,node_list,
            A_matrix,node_2_list,node_0_list,node_1,remove_edge_list = attack_node(G,A_matrix,time)
            A_matrix,recover_edge_list=recover_edge(G,A_matrix,node_2_list,node_0_list,node_1, time+1)
            attack_index.append(node_1)
            control_input_list.append([recover_edge_list[i]+remove_edge_list[i] for i in range(node_num)])
            time += 2
        else:
            control_input_list.append([0 for i in range(node_num)])
        # record
        state_list = []
        for i in range(node_num):
            state_list.append(node_list[i].state)
        state_list = np.array(state_list)
    state_recorder.append(state_list)
    plt.tight_layout()

    print("Nodes under attack:{}".format(attack_index))

    # Draw the evolution of node states
    plt.figure(figsize=(9,6))
    plt.xlabel('Time (sec)',font2)
    plt.ylabel('Agent State',font2)
    plt.xlim(xmax=iter_num * T, xmin=0)
    plt.ylim(ymax=20, ymin=0)
    state_recorder = np.array(state_recorder)
    y = np.array(state_recorder)

    sub_map = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')
    label1 = ['x' + str(i) +'(t)' for i in range(node_num)]

    is_first = True
    x = [T * i for i in range(iter_num+1)]
    for i in range(node_num):
        if(i in attack_index):
            x_ = [T * j for j in range(10 * attack_index.index(i) + 12)]
            plt.plot(x_, y[:10 * attack_index.index(i) + 12, i], color='k', label=label1[i].translate(sub_map), zorder=1)
            x_ = [T * j for j in range(10 * attack_index.index(i) + 11, iter_num+1)]
            plt.plot(x_, y[10 * attack_index.index(i) + 11:iter_num+1, i], color='k', linestyle="--", zorder=1)
        else:
            p = plt.plot(x, y[:, i], label=label1[i].translate(sub_map), zorder=2)
    for i in range(len(attack_index)):
        plt.scatter(T * (10 * i + 11), y[10 * i + 12, attack_index[i]], color='red', marker='x', s=100, zorder=3)
    plt.legend(loc='best',ncol=4)

    plt.title("The Consensus of MAS under Random Attacks",font1)
    plt.tight_layout()

    # plt.figure(figsize=(9, 6))
    # ax = plt.subplot(111)
    # plt.xlabel('Time Step', font2)
    # plt.ylabel('The Number of Control Input', font2)
    # plt.xlim(xmax=iter_num - 1, xmin=0)
    # x = [i for i in range(iter_num)]
    # y_ = np.array(control_input_list)
    #
    # cm = plt.get_cmap('gist_rainbow')
    # ax.set_prop_cycle('color', [cm(1. * i / node_num) for i in range(node_num)])
    #
    # for i in range(node_num):
    #     ax.plot(x, y_[:, i])
    #     ax.scatter(x, y_[:, i])
    # plt.title("The Number of Control Input under Attack Resilience", font2)
    # plt.tight_layout()

    plt.show()
