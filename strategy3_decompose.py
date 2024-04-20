import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import math
from sympy import Matrix
from matplotlib.pyplot import MultipleLocator

T = 0.1

class Node:
    Node_num = 0

    def __init__(self, index):
        self.index = index
        self.s_buffer=[]
        self.m_buffer=[]
        self.add_edge_list=[]
        self.remove_edge_list=[]
        self.is_stop=False
        self.parent_list=[]
        Node.Node_num += 1

    def system_decomposition(self,is_first):
        if(is_first):
            parent_list = get_parent(A_matrix, self.index)
            self.parent_list=list(parent_list)
        parent_list = get_parent(A_matrix, self.index)
        s_=[]
        for j in self.parent_list:
            s_=list(set(s_)|set(node_list[j].s_buffer))
        for j in parent_list:
            for boundary in s_:
                if (f(boundary[0]) == f(self.index) and f(j) != f(self.index)):
                    if ([j, self.index] not in self.remove_edge_list):
                        self.remove_edge_list.append([j, self.index])
                    if (boundary[0] not in parent_list and ([boundary[0], self.index] not in self.add_edge_list)):
                        self.add_edge_list.append([boundary[0], self.index])
                    node_list[boundary[0]].m_buffer.append([boundary[0], boundary[1]])
                else:
                    if(boundary not in self.s_buffer):
                        self.s_buffer.append(boundary)
            if (f(j) != f(self.index) and ((j, self.index) not in self.s_buffer)):
                self.s_buffer.append((j, self.index))
        for m in self.m_buffer:
            if (m not in self.remove_edge_list):
                self.remove_edge_list.append(list(m))
        self.m_buffer=[]
        for edge in self.add_edge_list:
            if (G.has_edge(edge[0], edge[1])==False):
                G.add_edge(edge[0],edge[1])
                add_edge_list.append(edge)
        for edge in self.remove_edge_list:
            if (G.has_edge(edge[0], edge[1])):
                G.remove_edge(edge[0], edge[1])
                remove_edge_list.append(edge)
        control_num=len(self.add_edge_list)+len(self.remove_edge_list)
        self.add_edge_list=[]
        self.remove_edge_list=[]
        return control_num

def f(node_index):
    for i in range(group_num):
        if(node_index in group_list[i]):
            return i

def generate_random_graph(n, m):
    while(True):
        node_list = []
        edge_list = []
        graph = {}
        for node in range(n):
            node_list.append(node)
        for node in node_list:
            graph[node] = []

        # choose root
        unassigned_node = [i for i in range(n)]
        assigned_node = []
        root = random.randint(0, n - 1)
        assigned_node.append(root)
        unassigned_node.remove(root)
        for node in node_list:
            if (node in set(unassigned_node)):
                index = random.randint(0, len(assigned_node) - 1)
                edge_list.append((assigned_node[index], node))
                graph[assigned_node[index]].append(node)
                assigned_node.append(node)
                unassigned_node.remove(node)

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
    return graph, edge_list

def decomposition(node_num,group_num):
    node_list = range(node_num)
    group_list=[]
    select_list=[]
    rest_list=range(node_num)
    rest_num=group_num
    for i in range(group_num-1):
        k = random.randint(1,int(2*len(rest_list)/rest_num-1))
        select_list = random.sample(rest_list, k)
        group_list.append(select_list)
        rest_list = list(set(rest_list).difference(set(select_list)))
    group_list.append(rest_list)
    return group_list

def get_boundary_num():
    boundary_list=[]
    for edge in G.edges:
        for i in range(group_num):
            if (edge[0] in group_list[i] and edge[1] not in group_list[i]):
                boundary_list.append(edge)
    return len(boundary_list)

def get_parent(A_matrix, index):
    parent_node = []
    for i in range(A_matrix.shape[0]):
        if (A_matrix[i][index] > 0):
            parent_node.append(i)
    return parent_node

def get_grandparent(A_matrix, index):
    grandparent_node = []
    parent_node = get_parent(A_matrix, index)
    for parent in parent_node:
        for i in range(A_matrix.shape[0]):
            if (A_matrix[i][parent] > 0):
                grandparent_node.append(i)
    return grandparent_node

def get_chlid(A_matrix, index):
    child_node = []
    for i in range(A_matrix.shape[0]):
        if (A_matrix[index][i] > 0):
            child_node.append(i)
    return child_node

def check_connection(edge_list,node_1_list,node_2_list):
    for edge in edge_list:
        if(edge[0] in node_1_list and edge[1] in node_2_list)or(edge[0] in node_2_list and edge[1] in node_1_list):
            return True
    return False

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
        now_color = color_rgb_start
        color_map.append(RGB_list_to_Hex(now_color))
        for color_index in range(1, color_sub_count):
            now_color = [now_color[0] + r_step, now_color[1] + g_step, now_color[2] + b_step]
            color_map.append(RGB_list_to_Hex(now_color))
        color_index_start = color_index_end
    return color_map

if __name__ == '__main__':
    # Drawing parameters
    color=['#5584AC','#95D1CC','#FAFFAF']
    color_ = ['#251D3A', '#E8F9FD']
    is_label = True
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 20, }
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 16, }
    font3 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 16, }

    # Topology parameters
    node_num =30
    tau = 1.2
    edeg_num = int(node_num * tau)
    group_num=3

    # Generate classification
    graph, edge_list = generate_random_graph(node_num, edeg_num)
    group_list = decomposition(node_num,group_num)
    G = nx.DiGraph(graph)

    # Draw original graph
    plt.figure(figsize=(12, 4))
    plt.subplot(1,3,1)
    plt.title("The original graph",font1,y=-0.05)
    plt.tight_layout()
    pos = nx.kamada_kawai_layout(G)
    nx.draw(G, pos=pos, with_labels=is_label, node_color=color[0], edge_color='lightgray')
    for i in range(group_num):
        nx.draw_networkx_nodes(G, pos=pos, nodelist=group_list[i], node_color=color[i])
    A_matrix = nx.adj_matrix(G).toarray()

    node_list = []
    for i in range(node_num):
        node = Node(i)
        node_list.append(node)
    trigger_node=random.randint(0, node_num-1)
    send_list = [trigger_node]
    receive_list = []
    triggerred_node_list = [trigger_node]
    buff_list = []
    control_input_list=[]
    memery_list=[]
    count = 1
    add_edge_list=[]
    remove_edge_list=[]
    plt.subplot2grid((1, 36), (0, 12),colspan=10)
    plt.title("The propagation of triggering signal",font1, y=-0.05, x=0.6)
    nx.draw(G, pos=pos, with_labels=is_label, node_color=color[1], edge_color='lightgray')
    iter_num=16

    for j in range(iter_num):
        control_input=[]
        A_matrix=nx.adj_matrix(G).toarray()
        memery_list.append(send_list)
        buff_list.append(get_boundary_num())
        for node in node_list:
            if (node.index in triggerred_node_list):
                is_first=False
                if(node.index in send_list):
                    is_first=True
                control_input.append(node.system_decomposition(is_first))
            else:
                control_input.append(0)
        control_input_list.append(list(control_input))
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
        if(len(receive_list)>0):
            count += 1
        send_list = receive_list
        receive_list = []

    for edge in G.edges:
        for i in range(group_num):
            if (edge[0] in group_list[i] and edge[1] not in group_list[i]):
                print(edge)
    colors = gradient_color(color_,color_sum=count)
    for i in range(count):
        nx.draw_networkx_nodes(G, pos=pos, nodelist=memery_list[i], node_color=colors[i])
    nx.draw_networkx_nodes(G, pos=pos, nodelist=[trigger_node], node_color='red')

    nx.draw_networkx_edges(G, pos=pos, edgelist=add_edge_list, edge_color='green')
    nx.draw_networkx_edges(G, pos=pos, edgelist=remove_edge_list, edge_color='red')

    cax = plt.subplot2grid((8, 72), (2, 45),rowspan=4,colspan=1)
    for i in range(count):
        if (i == 0):
            cax.fill_between([0, 1], i, i + 1, color='red')
        else:
            cax.fill_between([0, 1], i, i + 1, color=colors[i - 1])

    cax.set_ylim(0, count)
    cax.set_xticks([])
    cax.set_yticks(list(np.arange(0.5, count, 1)))
    cax.set_yticklabels(range(count))
    cax.yaxis.tick_left()
    plt.text(1.5, 0.8, "Trigger Sequence",font3, rotation=90)

    plt.subplot(1,3,3)
    plt.title("The reconfigured graph",font1, y=-0.05)
    plt.tight_layout()
    G.add_edges_from([[group_list[0][0],group_list[1][0]],[group_list[0][0],group_list[2][0]],[group_list[1][0],group_list[2][0]]])

    pos_0 = nx.kamada_kawai_layout(G)
    G.remove_edges_from([[group_list[0][0],group_list[1][0]],[group_list[0][0],group_list[2][0]],[group_list[1][0],group_list[2][0]]])
    nx.draw(G, pos=pos_0, with_labels=is_label, node_color=color[0], edge_color='lightgray')
    for i in range(group_num):
        nx.draw_networkx_nodes(G, pos=pos_0, nodelist=group_list[i], node_color=color[i])

    G = nx.DiGraph(graph)
    A_matrix = nx.adj_matrix(G).toarray()
    pos = nx.kamada_kawai_layout(G)
    node_list = []
    for i in range(node_num):
        node = Node(i)
        node_list.append(node)

    send_list = [trigger_node]
    receive_list = []
    triggerred_node_list = [trigger_node]


    sub_map = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')
    label1 = [r"$v_{%s}$"%str(i) for i in range(node_num)]

    plt.figure(figsize=(12, 6))
    ax=plt.subplot(121)
    plt.xlabel('Time Step',font2)
    plt.ylabel('The Number of Control Input',font2)
    plt.xlim(xmax=iter_num-1, xmin=0)
    x = [i for i in range(iter_num)]
    y_ = np.array(control_input_list)

    cm = plt.get_cmap('gist_rainbow')
    ax.set_prop_cycle('color', [cm(1. * i / node_num) for i in range(node_num)])

    for i in range(node_num):
        ax.plot(x, y_[:, i])
        ax.scatter(x, y_[:, i],label=label1[i])
    plt.legend(loc='best',ncol=3)
    plt.title("The Trend of the Number of Control Input",font1)
    plt.tight_layout()

    plt.subplot(122)
    plt.xlabel('Time Step',font2)
    plt.ylabel('The Number of Boundaries',font2)
    plt.xlim(xmax=iter_num-1, xmin=0)
    y = np.array(buff_list)
    plt.plot(x, y)
    plt.scatter(x, y)
    plt.title("The Trend of the Number of Boundaries",font1)
    plt.tight_layout()

    plt.figure(figsize=(12, 9))
    for i in range(8):
        add_edge_list = []
        remove_edge_list = []
        plt.subplot(3, 3, i+1)
        plt.title("k=%s" %i,font1, y=-0.1)
        plt.tight_layout()
        A_matrix=nx.adj_matrix(G).toarray()
        for node in node_list:
            if (node.index in triggerred_node_list):
                is_first=False
                if(node.index in send_list):
                    is_first=True
                node.system_decomposition(is_first)

        G.add_edges_from([[group_list[0][0], group_list[1][0]], [group_list[0][0], group_list[2][0]],
                          [group_list[1][0], group_list[2][0]]])
        pos = nx.kamada_kawai_layout(G)
        G.remove_edges_from([[group_list[0][0], group_list[1][0]], [group_list[0][0], group_list[2][0]],
                             [group_list[1][0], group_list[2][0]]])
        nx.draw(G, pos=pos, with_labels=True, node_color=color[1], edge_color='lightgray')
        for j in range(group_num):
            nx.draw_networkx_nodes(G, pos=pos, nodelist=group_list[j], node_color=color[j])
        if (i== 0):
            nx.draw_networkx_nodes(G, pos=pos, nodelist=send_list, node_color="red")
        elif(i<count):
            nx.draw_networkx_nodes(G, pos=pos, nodelist=send_list, node_color=colors[i])
        nx.draw_networkx_edges(G, pos=pos, edgelist=add_edge_list, edge_color='green')
        nx.draw_networkx_edges(G, pos=pos, edgelist=remove_edge_list, edge_color='red')
        for index in send_list:
            for k in range(A_matrix.shape[0]):
                if (A_matrix[index][k] > 0):
                    receive_list.append(k)
        receive_list = list(set(receive_list))
        for node in list(receive_list):
            if (node in triggerred_node_list):
                receive_list.remove(node)
            else:
                triggerred_node_list.append(node)
        send_list = receive_list
        receive_list = []
    plt.subplot(3, 3, 9)
    plt.title("k=%s" % 8,font1, y=-0.05)
    plt.tight_layout()
    G.add_edges_from([[group_list[0][0], group_list[1][0]], [group_list[0][0], group_list[2][0]],
                      [group_list[1][0], group_list[2][0]]])

    # pos = nx.kamada_kawai_layout(G)
    G.remove_edges_from([[group_list[0][0], group_list[1][0]], [group_list[0][0], group_list[2][0]],
                         [group_list[1][0], group_list[2][0]]])

    nx.draw(G, pos=pos_0, with_labels=True, node_color=color[1], edge_color='lightgray')
    for j in range(group_num):
        nx.draw_networkx_nodes(G, pos=pos_0, nodelist=group_list[j], node_color=color[j])

    plt.show()
