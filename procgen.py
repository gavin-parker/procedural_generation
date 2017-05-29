import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import matplotlib
import random
import cProfile
import multiprocessing

pr = cProfile.Profile()
pr.enable()

fig = matplotlib.pyplot.gcf()
fig.set_size_inches(10, 10)


def generate_voronoi(count):
    a = np.random.uniform(0, 100, count)
    b = np.random.uniform(0, 100, count)
    points = np.column_stack((a, b))
    vor = Voronoi(points)
    return vor


def smooth(vor, iterations=0):
    newPoints = []
    for region in vor.regions:
        points = []
        for idx in region:
            vert = vor.vertices[idx]
            points.append(vert)
        avg = np.mean(points, axis=0)
        if not np.isnan(avg).any():
            if 100 > avg[0] > 0 and 100 > avg[1] > 0:
                newPoints.append(avg)
    if iterations > 0:
        return smooth(Voronoi(np.vstack(newPoints)), iterations - 1)
    return Voronoi(np.vstack(newPoints))


vor = generate_voronoi(10000)
vor = smooth(vor, 1)


class Node:
    def __init__(self, idx):
        self.idx = idx
        self.neighbours = set([])
        self.height = 0

    def addEdge(self, neighbour):
        self.neighbours.add(neighbour)


def create_graph(vor):
    graph = {}
    for edge in vor.ridge_points:
        a = edge[0]
        b = edge[1]
        if a not in graph:
            graph[a] = Node(a)
        if b not in graph:
            graph[b] = Node(b)
        graph[a].addEdge(graph[b])
        graph[b].addEdge(graph[a])
    return graph


visited = set([])
queue = []


def create_mountain(vor, decrement, graph, sharpness):
    visited.clear()
    queue.clear()
    start = random.choice(list(graph.keys()))
    graph[start].height = 1.0
    visited.add(graph[start])
    queue.append(graph[start])
    create_slope(vor, decrement, graph[start], sharpness)


def create_slope(vor, decrement, node, sharpness):
    while len(queue) > 0:
        current = queue.pop(0)
        if current.height > 0.1:
            for neighbour in current.neighbours:
                if neighbour not in visited:
                    mod = np.random.uniform(0.9, 1.1) * sharpness;
                    neighbour.height += current.height * decrement * mod
                    if neighbour.height > 1:
                        neighbour.height = 1
                    visited.add(neighbour)
                    queue.append(neighbour)


graph = create_graph(vor)
for i in range(0, 10):
    create_mountain(vor, 0.74, graph, 0.9)


def lerp(start, end, t):
    return start + (end - start) * t


def lerpVec3(start, end, t):
    return [lerp(start[0], end[0], t), lerp(start[1], end[1], t), lerp(start[2], end[2], t)]


green = [220 / 256, 247 / 256, 99 / 256]
brown = [153 / 256, 134 / 256, 80 / 256]
lerpVec3(green, brown, 0.5)
voronoi_plot_2d(vor, show_points=False, show_vertices=False)

for key in graph.keys():
    node = graph[key]
    region = vor.regions[vor.point_region[node.idx]]
    if not -1 in region:
        polygon = [vor.vertices[i] for i in region]
        col = (tuple(lerpVec3(green, brown, node.height)))
        if node.height < 0.2:
            col = (0, 0, 1)
        plt.fill(*zip(*polygon), color=col)
pr.disable()
pr.print_stats(sort='cumtime')
plt.ylim((0, 100))
plt.xlim((0, 100))
plt.axis('off')
plt.savefig("test.png", bbox_inches='tight')
plt.show()
