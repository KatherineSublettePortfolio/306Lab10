import numpy as np
from heapq import heappush, heappop, heapify
from animation import draw
import argparse
import msvcrt as m
import json

class Node():
    """
    cost_from_start - the cost of reaching this node from the starting node
    state - the state (row,col)
    parent - the parent node of this node, default as None
    """
    def __init__(self, state, cost_from_start, parent = None):
        self.state = state
        self.parent = parent
        self.cost_from_start = cost_from_start


class EightPuzzle():
    
    def __init__(self, start_state, goal_state, method, algorithm, array_index):
        self.start_state = start_state
        self.goal_state = goal_state
        self.visited = [] # state
        self.method = method
        self.algorithm = algorithm
        self.m, self.n = start_state.shape 
        self.array_index = array_index
        

    def goal_test(self, current_state):
        # your code goes here:
        if np.array_equal(self.goal_state, current_state):
            return True

    def get_cost(self, current_state, next_state):
        # your code goes here:
        return 1

    def get_successors(self, state):
        successors = []
        #generates the matrix 
        ident_array = [[1,-1, 0 ,0],[0,0,-1,1]]
        #
        blank_coord = np.where(state == 0)

        for n in range(0,4):
            blank_i = blank_coord[0][0]
            blank_j = blank_coord[1][0]
            i = ident_array[0][n] + blank_i
            j = ident_array[1][n] + blank_j
            if (i < 3 and i >=0 and j < 3 and j >=0):
                return_state = state.copy()
                return_state[i][j] = state[blank_i][blank_j]
                return_state[blank_i][blank_j] = state[i][j]
                successors.append(return_state)
        return successors

    # heuristics function
    def heuristics(self, state):
        # your code goes here:
        if self.method == "Hamming":
            misplace = 0
            for row in range(0,3):
                for col in range(0,3):
                    current = state[row][col]
                    correct = self.goal_state[row][col]
                    if current != correct:
                        misplace = misplace + 1
            
            # print('(hamming)Current heuristic value: ' + str(misplace))
            return misplace

        if self.method == "Manhattan":
            distanceSum = 0
            for row in range(0,3):
                for col in range(0,3):
                    current = state[row][col]
                    for r in range(0,3):
                        for c in range(0,3):
                            if current == self.goal_state[r][c]:
                                r_expected = r
                                c_expected = c
                                distanceR = row - r_expected
                                distanceC = col - c_expected
                                distanceSum += abs(distanceR) + abs(distanceC)

            # print('(manhatan)Current heuristic value: ' + str(distanceSum))
            return distanceSum

    # priority of node 
    def priority(self, node, state):
        # use if-else here to take care of different type of algorithms
        # your code goes here:
        if self.algorithm == 'AStar':
            return node.cost_from_start + self.heuristics(state)
        elif self.algorithm == 'Greedy':
            return self.heuristics(state)
        else:
            return node.cost_from_start
    
    # draw 
    def draw(self, node):
        path=[]
        while node.parent:
            path.append(node.state)
            node = node.parent
        path.append(self.start_state)

        draw(path[::-1], self.array_index, self.algorithm, self.method)

    # solve it
    def solve(self):
        # use one framework to merge all five algorithms.
        # !!! In A* algorithm, you only need to return the first solution. 
        #     The first solution is in general possibly not the best solution, however, in this eight puzzle, 
        #     we can prove that the first solution is the best solution. 
        # your code goes here:    
        print(self.algorithm)
        container = [] # node
        count = 1
        state = self.start_state.copy()
        current_node = Node(state, 0, None)
        self.visited.append(state)
        
        if self.algorithm == 'Depth-Limited-DFS':
            container.append(current_node)
        elif self.algorithm == 'BFS': 
            container.insert(0,current_node)
        elif self.algorithm == 'UCS': 
            heappush(container, (self.priority(current_node, state),count, current_node))
            count += 1
        elif self.algorithm == 'Greedy':
            heappush(container, (self.priority(current_node, state),count, current_node))
            count += 1
        elif self.algorithm == 'AStar':
            heappush(container, (self.priority(current_node, state),count, current_node))
            heapify(container)
            count += 1

        while container:
            #print("in container")
            # if one solution is found, call self.draw(current_node) to show and save the animation.
            if self.algorithm == 'Depth-Limited-DFS':
                current_node = container.pop()  
            elif self.algorithm == 'BFS':
                current_node = container.pop()
            elif self.algorithm == 'UCS':
                current_node = heappop(container)[2]
            elif self.algorithm == 'Greedy':
                current_node = heappop(container)[2]
            elif self.algorithm == 'AStar':
                current_node = heappop(container)[2]

            self.visited.append(current_node.state)

            # print('Current Node')
            # print(current_node)

            successors = self.get_successors(current_node.state)
            
            # print('Number of successors: ' + str(len(successors)))
            # print(successors)
            # m.getch()

            for next_state in successors:
                if(current_node.cost_from_start < 15) or (self.algorithm != 'Depth-Limited-DFS'): 
                    is_not_visited = True
                    for v_state in self.visited:
                        if np.array_equal(v_state, next_state):
                            is_not_visited = False
                            # print("is_not_visited(false)")
                            # print(is_not_visited)
                            # print(v_state)
                            # m.getch()
                            break

                    # print("next_state has been visitied: " + str(is_not_visited))
                    if is_not_visited == True:
                        # print("in if not visited")
                        self.visited.append(next_state)
                        next_cost = current_node.cost_from_start + self.get_cost(current_node.state, next_state)
                        next_node = Node(next_state,next_cost,current_node)
                        if self.goal_test(next_state):
                            print("passed goaltest")
                            self.draw(next_node)
                            return

                        if self.algorithm == 'Depth-Limited-DFS':
                            container.append(next_node)
                        elif self.algorithm == 'BFS': 
                            container.insert(0,next_node)
                        elif self.algorithm == 'UCS': 
                            heappush(container, (self.priority(next_node, next_state), count, next_node))
                            count += 1
                        elif self.algorithm == 'Greedy':
                            heappush(container, (self.priority(next_node, next_state), count, next_node))
                            count += 1
                        elif self.algorithm == 'AStar':
                            heappush(container, (self.priority(next_node, next_state), count, next_node))
                            count += 1
            print('Number of items in container: ' + str(len(container)))
            # print('curent state of the container:')
            # print(container)
            # m.getch()

                         
if __name__ == "__main__":
    
    goal = np.array([[1,2,3],[4,5,6],[7,8,0]])
    start_arrays = [np.array([[1,2,0],[3,4,6],[7,5,8]]),
                    np.array([[8,1,3],[4,0,2],[7,6,5]])]
    methods = ["Hamming", "Manhattan"]
    algorithms = ['Depth-Limited-DFS', 'BFS', 'UCS', 'Greedy', 'AStar']
    
    parser = argparse.ArgumentParser(description='eight puzzle')

    parser.add_argument('-array', dest='array_index', required = True, type = int, help='index of array')
    parser.add_argument('-method', dest='method_index', required = True, type = int, help='index of method')
    parser.add_argument('-algorithm', dest='algorithm_index', required = True, type = int, help='index of algorithm')

    args = parser.parse_args()

    # Example:
    # Run this in the terminal using array 0, method Hamming, algorithm AStar:
    #     python eight_puzzle.py -array 0 -method 0 -algorithm 4
    game = EightPuzzle(start_arrays[args.array_index], goal, methods[args.method_index], algorithms[args.algorithm_index], args.array_index)
    game.solve()