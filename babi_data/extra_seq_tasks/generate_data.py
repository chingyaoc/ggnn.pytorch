import numpy as np
import string
import random
from collections import defaultdict
import sys
import networkx as nx
import networkx.algorithms as nxalg

class Person(object):
    def __init__(self, name):
        self.name = name
        self.purse = []

    def __str__(self):
        return "Person(%s, %s)" % (self.name, np.sum(self.purse))

    def gets(self, amount, display=True):
        if display:
            print "%s gets %s" % (self.name, amount)
        self.purse.append(amount)

    def gives_everything(self, other_person):
        print "%s gives-everything-to %s" % (self.name, other_person.name)
        for amount in self.purse:
            other_person.gets(amount, display=False)
        self.purse = []

    def loses_one(self):
        if len(self.purse) == 0:  return

        amount_to_lose = random.choice(self.purse)
        self.purse.remove(amount_to_lose)
        print "%s loses %s" % (self.name, amount_to_lose)
        

def make_change(person, denominations):
    denominations = sorted(denominations, reverse=True)
    total_money = np.sum(person.purse)
    result = []
    for j in range(len(denominations)):
        num_of_j = total_money / denominations[j]
        result += num_of_j * [denominations[j]]
        total_money -= num_of_j * denominations[j]
    return result


def make_money_story(options):
    num_entities = options["num_entities"]
    people = [Person(name) for name in string.ascii_uppercase[:num_entities]]
    denominations = options["denominations"]

    statements = []
    for person in people:
        person.gets(random.choice(denominations))

    for t in range(options["num_timesteps"]):
        giver, receiver = random.choice(people), random.choice(people)

        if giver.name == receiver.name:  continue

        action_type = np.random.randint(3)  # change to 4 if handling input order

        if action_type == 0:
            giver.loses_one()

        elif action_type <= 2:
            receiver.gets(random.choice(denominations))

        elif action_type == 3:
            giver.gives_everything(receiver)

    return people


def generate_change_data(options):
    people = make_money_story(options)
        
    change_giver = random.choice(filter(lambda p: len(p.purse) > 0, people))
    change = make_change(change_giver, options["denominations"]) + ["<end>"]
    print "eval make-change %s\t%s" % (change_giver.name, ",".join([str(c) for c in change]))


def generate_purse_data(options):
    people = make_money_story(options)

    change_giver = random.choice(filter(lambda p: len(p.purse) > 0, people))
    change = change_giver.purse + ["<end>"]
    print "eval coins %s\t%s" % (change_giver.name, ",".join([str(c) for c in change]))
    

def generate_sorted_purse_data(options):
    people = make_money_story(options)

    change_giver = random.choice(filter(lambda p: len(p.purse) > 0, people))
    change = sorted(change_giver.purse, reverse=True) + ["<end>"]
    print "eval coins %s\t%s" % (change_giver.name, ",".join([str(c) for c in change]))

def generate_shortest_path_data(options):

    while True:
        num_nodes = options["num_entities"]
        g = nx.random_graphs.connected_watts_strogatz_graph(num_nodes, 3, .5)
        source, target = np.random.randint(num_nodes, size=2)

        if source == target:  continue  # reject trivial paths

        paths = list(nxalg.all_shortest_paths(g, source=source, target=target))

        if len(paths) > 1:  continue  # reject when more than one shortest path

        path = paths[0]

        break

    for edge in g.edges():
        print "%s connected-to %s" % (edge[0], edge[1])
        print "%s connected-to %s" % (edge[1], edge[0])

    print "eval shortest-path %s %s\t%s" % (source, target, ",".join([str(v) for v in path]))
    

def generate_eulerian_circuit_data(options):

    while True:
        num_nodes = options["num_entities"]
        g = nx.random_regular_graph(2, num_nodes)

        try:
            path = list(nxalg.eulerian_circuit(g))
        except:
            continue

        # print path

        break

    for edge in g.edges():
        print "%s connected-to %s" % (edge[0], edge[1])
        print "%s connected-to %s" % (edge[1], edge[0])

    first_edge = path[0]

    node_list = [str(edge[0]) for edge in path]
    print "eval eulerian-circuit %s %s\t%s" % (first_edge[0], first_edge[1],
                                               ",".join(node_list))


##################### noisy data #######################

def _generate_random_node_index(n_nodes):
    idx = range(n_nodes)
    random.shuffle(idx)
    return idx

def _relabel_nodes_in_edges(edges, idx):
    """
    edges is a list of tuples
    """
    return [(idx[e[0]], idx[e[1]]) for e in edges]

def _relabel_nodes_in_path(path, idx):
    return [idx[n] for n in path]

def generate_noisy_shortest_path_data(options):

    while True:
        num_nodes = options["num_entities"]
        min_path_len = options["min_path_len"]
        g = nx.random_graphs.connected_watts_strogatz_graph(num_nodes, 3, .5)
        source, target = np.random.randint(num_nodes, size=2)

        if source == target:  continue  # reject trivial paths

        paths = list(nxalg.all_shortest_paths(g, source=source, target=target))

        if len(paths) > 1:  continue  # reject when more than one shortest path

        path = paths[0]

        if len(path) < min_path_len: continue   # reject paths that's too short

        break

    edges = g.edges()

    num_confusing = options['num_confusing']
    if num_confusing > 0:
        g_confusing = nx.random_graphs.connected_watts_strogatz_graph(num_confusing, 3, .5)

        for e in g_confusing.edges():
            edges.append((e[0] + num_nodes, e[1] + num_nodes))

        random.shuffle(edges)

    # randomize index
    idx = _generate_random_node_index(num_nodes + num_confusing)
    new_edges = _relabel_nodes_in_edges(edges, idx)
    new_path = _relabel_nodes_in_path(path, idx)

    for edge in new_edges:
        print "%s connected-to %s" % (edge[0], edge[1])
        print "%s connected-to %s" % (edge[1], edge[0])

    print "eval shortest-path %s %s\t%s" % (idx[source], idx[target], ",".join([str(v) for v in new_path]))
 

def generate_noisy_eulerian_circuit_data(options):
    """
    This is a noisy version of the eularian circuit problem.
    """
    while True:
        num_nodes = options["num_entities"]
        g = nx.random_regular_graph(2, num_nodes)

        try:
            path = list(nxalg.eulerian_circuit(g))
        except:
            continue
        break

    edges = g.edges()

    # generate another misleading cycle
    num_confusing = options["num_confusing"]
    if num_confusing > 0:
        g_confusing = nx.random_regular_graph(2, num_confusing)

        for e in g_confusing.edges():
            edges.append((e[0] + num_nodes, e[1] + num_nodes))

    random.shuffle(edges)

    # randomize index
    idx = _generate_random_node_index(num_nodes + num_confusing)
    new_edges = _relabel_nodes_in_edges(edges, idx)
    new_path = _relabel_nodes_in_edges(path, idx)

    for edge in new_edges:
        print "%s connected-to %s" % (edge[0], edge[1])
        print "%s connected-to %s" % (edge[1], edge[0])

    first_edge = new_path[0]

    node_list = [str(edge[0]) for edge in new_path]
    print "eval eulerian-circuit %s %s\t%s" % (first_edge[0], first_edge[1],
                                               ",".join(node_list))

def main(task, options):
    if task == 1:
        generate_change_data(options)

    elif task == 2:  # requires knowing input order
        generate_purse_data(options)

    elif task == 3:
        generate_sorted_purse_data(options)

    elif task == 4:
        # generate_shortest_path_data(options)
        generate_noisy_shortest_path_data(options)

    elif task == 5:
        # generate_eulerian_circuit_data(options)
        generate_noisy_eulerian_circuit_data(options)



if __name__ == "__main__":
    if len(sys.argv) < 4:
        print 'python generate_data.py <task-id> <num-entities> <num-examples> [<num-confusing-entities>]'
    else:
        task = int(sys.argv[1])
        num_entities = int(sys.argv[2])
        num_examples = int(sys.argv[3])
        num_confusing = int(sys.argv[4]) if len(sys.argv) >= 5 else 0

        if task <= 3:
            options = {
                "num_entities" : num_entities,
                "num_timesteps" : 20,
                "denominations" : [1, 5, 10, 25]
                }

        elif task >= 4:
            options = {
                "num_entities" : num_entities,
                "num_confusing" : num_confusing,
                "min_path_len" : 3
                }

        for i in xrange(num_examples):
            main(task, options)
