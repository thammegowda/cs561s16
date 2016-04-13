
# Tokens
QRY_SEP = "******"
NODE_SEP = "***"
GIVEN = "|"
TRUE = "+"
FALSE = "-"
JOINT = ","
EQUALS = "="
PROBABILITY = "P"
EXP_UTILITY = "EU"
MAX_EXP_UTILITY = "MEU"
NOR_NODE = "normal"
DEC_NODE = "decision"
UTILITY_NODE = "utility"


class Node(object):

    def __init__(self, name, cpt, _type='normal'):
        """
        Creates a node
        :param name: name for this node
        :param cpt: Conditional probability table
        :param _type: node type {'normal', 'decision', 'utility'}
        :return:
        """
        self._type = _type
        self.name = name
        self.cpt = cpt
        self.parents = {}

        # fist row in cpt is its header
        if len(cpt[0]) > 1:     # it got parents
            for p in cpt[0][1:]:
                self.parents[p] = None

    def num_parents(self):
        """
        gets number of parents
        :return: number of parents
        """
        return len(self.parents)

    def __repr__(self):
        return "Node (%s)" % self.name


class Query(object):

    def __init__(self, _type, query, given=None):
        self._type = type
        self.given = query
        self.query = given


def tokenize(statement):
    return statement.strip().split()


def parse_cpt(cpt):
    name = cpt[0][0]
    _type = UTILITY_NODE if name == UTILITY_NODE else UTILITY_NODE if cpt[1][0] == UTILITY_NODE else NOR_NODE
    return Node(name, cpt, _type)

def parse(in_file):
    qs = []
    nodes = {}

    with open(in_file) as reader:

        for line in reader:         # queries first
            tokens = tokenize(line)
            if len(tokens) == 0:    # skip empty lines
                continue
            if tokens[0] == QRY_SEP:
                break               # Done parsing queries
            qs.append(Query(MAX_EXP_UTILITY, tokens[0]))
        cpt = []
        for line in reader:         # parse nodes
            tokens = tokenize(line)
            if len(tokens) == 0:    # skip empty lines if there are any
                continue
            if len(tokens) == 1 and tokens[0] == NODE_SEP:
                if cpt:
                    node = parse_cpt(cpt)
                    nodes[node.name] = node
                cpt = []
            else:
                cpt.append(tokens)
        if cpt:
            node = parse_cpt(cpt)
            nodes[node.name] = node
    return qs, nodes

if __name__ == '__main__':
    qs, nodes = parse("/home/tg/work/coursework/cs561/csci561s16/hw3/tests/sample01.txt")

    print(qs)
    print(nodes)
