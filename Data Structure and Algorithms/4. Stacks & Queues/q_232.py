# https://leetcode.com/problems/implement-queue-using-stacks/



class MyQueue:
    def __init__(self):
        self.in_stack = []
        self.out_stack = []

    def push(self, x: int) -> None:
        self.in_stack.append(x)

    def pop(self) -> int:
        if self.empty():
            return None
        if self.out_stack is not None:
            return self.out_stack.pop()
        while len(self.in_stack) != 0:
            self.out_stack.append(self.in_stack.pop())
        return self.out_stack.pop()

    def peek(self) -> int:
        return self.out_stack[-1] if len(self.out_stack) != 0 else self.in_stack[0]

    def empty(self) -> bool:
        return len(self.in_stack) == 0 and len(self.out_stack) == 0
