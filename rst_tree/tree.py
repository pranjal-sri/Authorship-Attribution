class Tree:
    def __init__(self):
        self.start, self.end = -1, -1
        self.text = None
        self.left, self.right = None, None
        self.parent = None
        self.children = set()