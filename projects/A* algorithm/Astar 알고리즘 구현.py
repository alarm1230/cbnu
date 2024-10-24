import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import math

## 노드 클래스 ##
class Node:
    
    # 노드 클래스의 생성자 함수
    def __init__(self, x, y, value):
        self.x = x # 노드의 x좌표
        self.y = y # 노드의 y좌표
        self.value = value # 맵에서 장애물 존재 판단에 사용하는 값
        self.g = 10**2 # 출발 노드에서 현 노드까지 알려진 최소 거리, 초기 값은 매우 큰 값
        self.h = 10**2 # 현 노드에서 도착 노드까지 휴리스틱을 사용하여 계산된 거리. 초기 값은 매우 큰 값
        self.cost = 10**2 # 현 노드의 총 비용 : g + h = f <-
        self.parent = None # 현 노드의 부모 노드, Path 구성 시 사용됨.
    
    # A* 알고리즘 수행 전 해당 변수 값 초기화하는 함수
    def Reset_costs(self):
        self.g = 10**2
        self.h = 10**2
        self.cost = 10**2
        self.parent = None
    
    # 노드 클래스의 string 표현을 결정하는 함수    
    def __str__(self):
        return '({0}, {1})'.format(self.x, self.y) # self.x -> {0}, self.y -> {1}
    
    # 현 노드와 다른 노드와의 동일성을 판단하는 함수 (두 값이 같은지 비교)
    def __eq__(self, other):
        if self.__str__() == other.__str__():
            return True
        else:
            return False
        
    
## 엣지 클래스 ##

class Edge:

    # 엣지 클래스의 생성자 함수
    def __init__(self, node1, node2, cost):
        self.node1 = node1
        self.node2 = node2
        self.cost = cost # 노드 1에서 노드 2로 이동 시 발생 비용
        
    # 엣지 클래스의 string 표현을 결정하는 함수
    def __str__(self):
        return '(Node{0},Node{1})'.format(str(self.node1), str(self.node2))


    
## 그래프 클래스 ##
class Graph:
    
    # 그래프 클래스의 생성자 함수
    def __init__(self, grid):
        self.grid = grid # 2차원 np array 형식의 자료
        self.nodes = {} # 딕셔너리 형식의 자료, why 딕셔너리? -> 접근 시 시간 복잡도 적음
        self.edges = {} # 2차원(key1 : 노드1, key2 : 노드2, value : 엣지) 형식의 딕셔너리
        
        # 그리드 정보를 기반으로 노드 생성
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                new_node = Node(i, j, grid[i, j]) # numpy형태의 2차원 배열의 0,1 값을 갖고있음
                self.nodes[str(new_node)] = new_node
                
        #각 노드들의 인접노드를 연결하여 엣지 생성
        #인접노드가 장애물인 경우(노드의 value가 0이 아닌 경우) 엣지 비용에 큰 값을 할당
        for node1 in self.nodes.values():
            self.edges[str(node1)] = {} #key1: 노드1 string 표현
            for node2 in self.Get_neighborhood(node1): #노드 1의 인접노드들을 리스트 형태로 반환하는 함수
                cost = math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)
                if node1.value != 0 or node2.value != 0:
                    cost = 10 ** 10
                self.edges[str(node1)][str(node2)] = Edge(node1, node2, cost)
            
    # 해당 노드의 인접노드들을 리스트 형태로 변환
    def Get_neighborhood(self,node):
        neighborhood = []
        for i in range (-1, 2, 1): # i : x방향 + @ 증가
            for j in range (-1, 2, 1): # j : y방향 + @ 증가
                if (i != 0 or j != 0): # 현재 노드 배제 (x, y 방향 @ 증가분 모두 0)
                    if node.x + i >= 0 and node.x + i < self.grid.shape[0]: # x + @ 가 맵 안에 포함
                        if node.y + j >= 0 and node.y + j < self.grid.shape[1]: # y + @ 가 맵 안에 포함
                            neighborhood.append(self.Get_node(node.x + i, node.y + j)) # 그 인접 노드 리스트
        return neighborhood
                    
    # 주어진 (x,y) 위치 노드 반환
    def Get_node(self,x,y):
        return self.nodes[self.__make_node_id(x,y)]

    # 노드1, 노드2 로 구성된 엣지 반환
    def Get_edge(self, node1, node2):
        return self.edges[str(node1)][str(node2)]

    # 주어진 (x, y) 사용해 노드 id 생성
    def __make_node_id(self, x, y):
        return '({0}, {1})'.format(x,y)

    
#최적 경로 탐색을 수행하는 객체 생성을 위한 클래스
class Path_planner():
    
    #출발노드, 도착노드, 그래프 정보를 사용해 최적경로를 탐색하는 함수
    def Get_shortest_path(self, start_node, finish_node, graph):
        #그래프 정보 받아오기
        nodes = list(graph.nodes.values()) #딕셔너리 형태의 노드집합을 리스트로 정의
        edges = graph.edges #딕셔너리 형태의 엣지 집합
        
        # 노드 g, h, cost 정보 초기화
        for node in nodes:
            node.Reset_costs()
            
        #open, closed 리스트 준비
        open_nodes = []
        closed_nodes = []
        
        #출발 노드 준비 및 open 노드 리스트에 삽입
        start_node.g = 0
        start_node.h = self.Get_h_cost(start_node, finish_node)
        start_node.cost = start_node.g + start_node.h
        open_nodes.append(start_node)
        
        #도착 노드가 closed트노드 리스트에 포함될 때까지 수행
        while finish_node not in closed_nodes:
            #step 1. open 노드 리스트 내 최소 값 노드 선택
            open_nodes.sort(key = lambda node: node.cost) # sort : 정렬 - 무거움
            selected_node = open_nodes[0]
            
            #선택 노드의 이웃 노드 open(새롭게 열린 노드인 경우) 또는 update(이미 open 노드 리스트ㅔ 존재하는 경우)
            neighborhood = [node for node in graph.Get_neighborhood(selected_node) if node not in closed_nodes]
            
            for new_node in neighborhood: #이웃노드 개별 탐색
                new_g = selected_node.g + graph.Get_edge(selected_node, new_node).cost
                new_h = self.Get_h_cost(new_node, finish_node)
                new_cost = new_g + new_h
                
                if new_node in open_nodes: #현재 탐색중인 이웃노드가 open list에 이미 있는 경우
                    if new_g < new_node.g : #기존 cost보다 현재 찾은 cost값이 더 낮은 경우 g, cost, 부모정보 업데이트
                        new_node.g = new_g
                        new_node.cost = new_cost
                        new_node.parent = selected_node
                else: #현재 탐색중인 이웃 노드가 open리스트에 없는 경우 값 갱신 후 삽입
                    new_node.g = new_g
                    new_node.h = new_h
                    new_node.cost = new_cost
                    new_node.parent = selected_node
                    open_nodes.append(new_node)
            open_nodes.remove(selected_node) #open 노드 리스트에서 현 선택노드 제거
            closed_nodes.append(selected_node) #closed 노드 리스트에 현 선택노드 삽입
            
        return self.Make_path(finish_node, nodes) #부모노드 정보를 이용해 경로 생성
    

    #노드 1과 노드 2 사이의 h값 계산 후 반환
    def Get_h_cost(self, node1, node2):
        return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)

    #부모노드 정보를 활용하여 경로 생성
    def Make_path(self, finish_node, nodes):
        path = []
        current_node = finish_node #도착 노드에서부터 추적 시작
        while current_node.parent != None: #부모 노드가 존재하지 않을 때까지 수행
            path.append(current_node) #현재 노드를 경로에 포함
            current_node = current_node.parent #부모 노드를 현재 노드로 지정
        path.append(current_node) #현재노드(출발노드)를 경로에 포함
        path.reverse() #도착 -> 출발을 출발 -> 도착으로 순서 변환
        return path
    
    
    
### 그래프 생성 ###

# 맵데이터 불러오기
file_path = "Map.csv"
map_dataframe = pd.read_csv(file_path, index_col=0, header=0)
map = map_dataframe.values # dataframe -> np array 변환

graph = Graph(map)

# 출발 및 도착 노드 선정
start_node = graph.Get_node(1,1)
finish_node = graph.Get_node(24, 24)

# 최단 거리 경로 탐색
shortest_path = Path_planner().Get_shortest_path(start_node, finish_node, graph)

### 최단 거리 경로 및 맵 가시화###
for node in shortest_path:
    print(f"({node.x}, {node.y})")
    map[node.x, node.y] = 2
cmap = colors.ListedColormap(['white', 'black', 'blue'])
norm = colors.Normalize(vmin=0, vmax=2)
fig, ax = plt.subplots()
ax.imshow(map, cmap = cmap, norm = norm)

plt.show()