class MaxIterationException(Exception):
    def __init__(self):
        super().__init__(f"Failed to converge after max_iter iterations.")