import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import math
from matplotlib.pyplot import MultipleLocator

T=0.1 #time step

class Node:
    node_num=0
    def __init__(self,index,state):
        self.index=index
        self.state=state
        self.neighbor=[]
        self.neighbor_info=[]
        self.remove_edge_list=[]
        Node.node_num+=1

    def get_info(self,A_matrix,state_list):
        state_neighbor=0
        for i in range (Node.node_num):
            if(A_matrix[i][self.index]>0):
                state_neighbor += (state_list[i] - self.state)
        self.state=self.state+state_neighbor*T

    def optimization_simplify(self,A_matrix,G):
        A_matrix_ = np.array(A_matrix, copy=True)
        child_list=get_chlid(A_matrix_,self.index)
        parent_list = get_parent(A_matrix_, self.index)
        for node_1 in parent_list:
            node_1_child_list=get_chlid(A_matrix_,node_1)
            indegree = len(set(node_1_child_list) & set(parent_list))
            outdegree= len(set(node_1_child_list) & set(child_list))
            if(indegree>outdegree):
                self.remove_edge_list.append([node_1,self.index])
                G.remove_edge(node_1, self.index)
                nx.draw_networkx_edges(G, pos=pos, edgelist=[(node_1, self.index)], edge_color='r', style='dashed')
        control_num=len(self.remove_edge_list)
        self.remove_edge_list=[]
        return control_num

def generate_random_graph(n, m):
    while(True):
        node_list = []
        edge_list = []
        graph = {}
        for node in range(n):
            node_list.append(node)
        for node in node_list:
            graph[node] = []  # Graph is a dict structure, initialized as an empty list

        # choose root
        unassigned_node = [i for i in range(n)]
        assigned_node = []
        root = random.randint(0, n - 1)
        assigned_node.append(root)
        unassigned_node.remove(root)
        for node in node_list:  # Create a directed spanning tree
            if (node in set(unassigned_node)):
                index = random.randint(0, len(assigned_node) - 1)
                edge_list.append((assigned_node[index], node))
                graph[assigned_node[index]].append(node)
                assigned_node.append(node)
                unassigned_node.remove(node)

        # Connect to root
        unassigned_node = [i for i in range(n)]
        assigned_node = []
        assigned_node.append(root)
        unassigned_node.remove(root)
        for node in node_list:
            if (node in set(unassigned_node)):
                index = random.randint(0, len(assigned_node) - 1)
                edge_list.append((node, assigned_node[index]))
                graph[node].append(assigned_node[index])
                assigned_node.append(node)
                unassigned_node.remove(node)
        add_edge_num = m - 2 * (n - 1)
        if (add_edge_num > 0):
            count = 0
            while (True):
                node_set = [i for i in range(n)]
                index_a = random.randint(0,n - 1)
                node_set.remove(index_a)
                index_ = random.randint(0, n - 2)
                index_b = node_set[index_]
                if index_b not in graph[index_a]:
                    edge_list.append((index_a, index_b))
                    graph[index_a].append(index_b)
                    count += 1
                if (count >= add_edge_num):
                    break
        G = nx.DiGraph(graph)
        sign=True
        for i in range(n):
            for j in range(n):
                if(nx.has_path(G, source=i,target=j)==False):
                    sign=False
        if(sign):
            break
    return graph

def get_parent(A_matrix,index):
    parent_node=[]
    for i in range(A_matrix.shape[0]):
        if(A_matrix[i][index]>0):
            parent_node.append(i)
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
    return child_node

def get_outdegree(A_matrix):
    return np.sum(A_matrix,axis=1)

def get_indegree(A_matrix):
    return np.sum(A_matrix,axis=0)

def get_degree(A_matrix):
    return np.sum(A_matrix,axis=0)+np.sum(A_matrix,axis=1)

# RGB format color conversion to hexadecimal color format
def RGB_list_to_Hex(RGB):
    color = '#'
    for i in RGB:
        num = int(i)
        color += str(hex(num))[-2:].replace('x', '0').upper()
    return color

# Hexadecimal color format color conversion to RGB format
def Hex_to_RGB(hex):
    r = int(hex[1:3], 16)
    g = int(hex[3:5], 16)
    b = int(hex[5:7], 16)
    rgb = str(r) + ',' + str(g) + ',' + str(b)
    return rgb, [r, g, b]

# Generate gradient colors
def gradient_color(color_list, color_sum=10):
    color_center_count = len(color_list)
    color_sub_count = int(color_sum / (color_center_count - 1))
    color_index_start = 0
    color_map = []
    for color_index_end in range(1, color_center_count):
        color_rgb_start = Hex_to_RGB(color_list[color_index_start])[1]
        color_rgb_end = Hex_to_RGB(color_list[color_index_end])[1]
        r_step = (color_rgb_end[0] - color_rgb_start[0]) / color_sub_count
        g_step = (color_rgb_end[1] - color_rgb_start[1]) / color_sub_count
        b_step = (color_rgb_end[2] - color_rgb_start[2]) / color_sub_count
        # 生成中间渐变色
        now_color = color_rgb_start
        color_map.append(RGB_list_to_Hex(now_color))
        for color_index in range(1, color_sub_count):
            now_color = [now_color[0] + r_step, now_color[1] + g_step, now_color[2] + b_step]
            color_map.append(RGB_list_to_Hex(now_color))
        color_index_start = color_index_end
    return color_map

if __name__ == '__main__':
    # Drawing parameters
    color = ['#557B83','#39AEA9', '#A2D5AB', '#E5EFC1', 'green']
    is_label=False
    plt.rcParams['font.serif'] = ['Times New Roman']

    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 18, }

    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 16, }

    font3 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 14, }

    # Topology parameters
    node_num = 20
    tau = 2
    edge_num = int(2*node_num * tau)

    # Draw original graph
    plt.figure(figsize=(12, 4))
    plt.subplot2grid((1, 3), (0, 0))
    plt.tight_layout()
    plt.title("The original graph",font1,y=-0.05)
    graph = generate_random_graph(node_num, edge_num)
    G = nx.DiGraph(graph)
    pos = nx.kamada_kawai_layout(G)
    nx.draw(G, pos=pos,with_labels=is_label,node_color=color[1],edge_color='lightgray')


    node_set = [i for i in range(node_num)]
    A_matrix = nx.adj_matrix(G).toarray() # adjacency matrix

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

    # Draw reconfiguration process
    plt.subplot2grid((1, 36), (0, 12),colspan=10)
    plt.tight_layout()
    plt.title("The propagation of triggering signal",font1,y=-0.05,x=0.6)
    nx.draw(G, pos=pos, with_labels=is_label, node_color=color[1], edge_color='lightgray')
    trigger_node=random.randint(0, node_num-1)
    send_list = [trigger_node]
    receive_list = []
    triggerred_node_list = [trigger_node]
    buff_list = []
    memery_list=[]
    count = 1
    num=1
    iter_num=16
    control_input_list = []

    #The propagation of triggering signal (Until all nodes are triggered)
    while (True):
        control_input = []
        memery_list.append(send_list)
        for node in node_list:
            if(node.index in triggerred_node_list):
                control_input.append(node.optimization_simplify(A_matrix,G))
            else:
                control_input.append(0)
        control_input_list.append(list(control_input))
        A_matrix = nx.adj_matrix(G).toarray()
        buff_list.append(list(get_indegree(A_matrix)))
        for index in send_list:
            for i in range(A_matrix.shape[0]):
                if (A_matrix[index][i] > 0):
                    receive_list.append(i)
        receive_list = list(set(receive_list))
        for node in list(receive_list):
            if (node in triggerred_node_list):
                receive_list.remove(node)
            else:
                triggerred_node_list.append(node)
        if(len(receive_list)==0):
            num=count
            break
        count += 1
        send_list = list(receive_list)
        receive_list = []
    # Until the number of iterations is reached
    while (True):
        control_input = []
        for node in node_list:
            if(node.index in triggerred_node_list):
                control_input.append(node.optimization_simplify(A_matrix, G))
            else:
                control_input.append(0)
        control_input_list.append(list(control_input))
        A_matrix = nx.adj_matrix(G).toarray()
        buff_list.append(list(get_indegree(A_matrix)))
        count+=1
        if (count>=iter_num):
            break

    sub_map = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')
    label1 = [r"$v_{%s}$" % str(i) for i in range(node_num)]

    color_ = ['#251D3A', '#E8F9FD']
    colors = gradient_color(color_,color_sum=len(memery_list))
    for i in range(len(memery_list)):
        if(i==0):
            nx.draw_networkx_nodes(G, pos=pos, nodelist=memery_list[i], node_color='red')
        else:
            nx.draw_networkx_nodes(G, pos=pos, nodelist=memery_list[i], node_color=colors[i])

    cax = plt.subplot2grid((8, 72), (2, 45), rowspan=4, colspan=1)
    for i in range(len(memery_list)):
        if (i == 0):
            cax.fill_between([0, 1], i, i + 1, color='red')
        else:
            cax.fill_between([0, 1], i, i + 1, color=colors[i])
    cax.set_ylim(0, len(memery_list))
    cax.set_xticks([])
    cax.set_yticks(list(np.arange(0.5, len(memery_list), 1)))
    cax.set_yticklabels(range(len(memery_list) ))
    cax.yaxis.tick_left()
    plt.text(1.5, 0.8, "Trigger Sequence",font3, rotation=90)

    # Draw reconfiguration result
    plt.subplot2grid((1, 3), (0, 2))
    plt.title("The reconfigured graph",font1, y=-0.05)
    pos = nx.kamada_kawai_layout(G)
    nx.draw(G, pos=pos, with_labels=is_label, node_color=color[1], edge_color='lightgray')
    plt.tight_layout()

    #Draw the trend of Node Indegree
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.xlim(xmax=count-1, xmin=0)
    x = list(range(count))
    y1 = np.array(buff_list)
    for i in range(node_num):
        if(y1[0, i]>y1[count-1, i]):
            plt.plot(x, y1[:, i])
            plt.scatter(x, y1[:, i],label=label1[i])
    t = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(t)
    plt.title("The Trend of Node Indegree in Network Simplification",font2)
    plt.xlabel('Time Step',font2)
    plt.ylabel('Node Indegree',font2)
    plt.legend(loc='best',ncol=3)

    # Draw the trend of average Node Indegree
    plt.subplot(122)
    z1=np.average(np.array(buff_list),axis=1)
    plt.xlim(xmax=count-1, xmin=0)
    plt.plot(x, z1,label=r"$\bar{d}^{in}\left ( G \right ) $")
    plt.scatter(x, z1)
    t = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(t)
    plt.title("The Trend of Average Indegree in Network Simplification",font2)
    plt.xlabel('Time Step',font2)
    plt.ylabel('Average Indegree',font2)
    plt.legend(loc='best')

    # Draw the detail of reconfiguration
    G=nx.DiGraph(graph)
    A_matrix = nx.adj_matrix(G).toarray()
    pos = nx.kamada_kawai_layout(G)
    node_list=[]
    for i in range(node_num):
        node=Node(i,0)
        node_list.append(node)
    send_list = [trigger_node]
    receive_list = []
    triggerred_node_list = [trigger_node]
    memery_list=[]
    plt.figure(figsize=(12, 8))
    count=1
    while (True):
        if(count<=6):
            plt.subplot(2,3,count)
            plt.title("k=%s"%(count-1),font1,y=-0.05)
            plt.tight_layout()
            nx.draw(G, pos=pos, with_labels=True, node_color=color[1], edge_color='lightgray')
            if(count==1):
                nx.draw_networkx_nodes(G, pos=pos, nodelist=send_list, node_color="red")
            else:
                nx.draw_networkx_nodes(G, pos=pos, nodelist=send_list, node_color=colors[count-1])
        memery_list.append(send_list)
        for node in node_list:
            if (node.index in send_list):
                node.optimization_simplify(A_matrix,G)
        A_matrix = nx.adj_matrix(G).toarray()  # 邻接矩阵
        count += 1
        for index in send_list:
            for i in range(A_matrix.shape[0]):
                if (A_matrix[index][i] > 0):
                    receive_list.append(i)
        receive_list = list(set(receive_list))
        for node in list(receive_list):
            if (node in triggerred_node_list):
                receive_list.remove(node)
            else:
                triggerred_node_list.append(node)
        if(len(receive_list)==0):
            break
        send_list = receive_list
        receive_list = []
    if(count<=6):
        plt.subplot(2, 3, 6)
        plt.title("k=5",font1,y=-0.05)
        pos = nx.kamada_kawai_layout(G)
        nx.draw(G, pos=pos, with_labels=True, node_color=color[1], edge_color='lightgray')
        plt.tight_layout()

    # plt.figure(figsize=(9, 6))
    # ax = plt.subplot(111)
    # plt.xlabel('Time Step', font2)
    # plt.ylabel('The Number of Control Input', font2)
    # plt.xlim(xmax=iter_num - 1, xmin=0)
    # x = [i for i in range(iter_num)]
    # y_ = np.array(control_input_list)
    # cm = plt.get_cmap('gist_rainbow')
    # ax.set_prop_cycle('color', [cm(1. * i / node_num) for i in range(node_num)])
    # for i in range(node_num):
    #     ax.plot(x, y_[:, i])
    #     ax.scatter(x, y_[:, i])
    # plt.title("The Number of Control Input under Network Simplification", font2)
    # plt.tight_layout()

    plt.show()
