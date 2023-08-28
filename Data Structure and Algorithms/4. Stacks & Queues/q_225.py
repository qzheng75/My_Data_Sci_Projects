# https://leetcode.com/problems/implement-stack-using-queues/


class MyStack:
    def __init__(self):
        self.queue = []

    def push(self, x: int) -> None:
        size = len(self.queue)
        self.queue.append(x)
        while size > 0:
            self.queue.append(self.queue.pop(0))
            size -= 1

    def pop(self) -> int:
        return self.queue.pop(0)

    def top(self) -> int:
        return self.queue[0]

    def empty(self) -> bool:
        return len(self.queue) == 0