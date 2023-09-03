from typing import List
import random
import heapq
from collections import deque


class Sorting:
    @staticmethod
    def supports_comparison(obj):
        return all(hasattr(obj, dunder) for dunder in ("__eq__", "__lt__", "__le__", "__gt__", "__ge__"))

    @staticmethod
    def is_homogeneous(arr):
        if not arr:
            return True
        first_element_type = type(arr[0])
        for element in arr:
            if type(element) != first_element_type:
                return False
        return True
    
    @staticmethod
    def type_checking(arr):
        if not arr or len(arr) == 0:
            return
        assert Sorting.is_homogeneous(arr), 'Elements in the array to sort must be homogenous in types.'
        assert Sorting.supports_comparison(arr[0]), 'Elements in the array must be of type that supports comparison.'

    @staticmethod
    def bubble_sort(arr: List) -> List:
        if not arr or len(arr) <= 1:
            return arr
        Sorting.type_checking(arr)
        end = len(arr) - 1
        start = 0
        last_swap = start
        while start < end:
            last_swap = start
            for j in range(start, end):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    last_swap = j
            end = last_swap
        return arr
    
    @staticmethod
    def cocktail_shaker_sort(arr: List) -> List:
        if not arr or len(arr) <= 1:
            return arr
        Sorting.type_checking(arr)
        start, end = 0, len(arr) - 1
        last_swap = start
        while start < end:
            last_swap = start
            for j in range(start, end):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    last_swap = j
            end = last_swap
            for j in range(end, start, -1):
                if arr[j] < arr[j - 1]:
                    arr[j], arr[j - 1] = arr[j - 1], arr[j]
                    last_swap = j
            start = last_swap
        return arr
    
    @staticmethod
    def insertion_sort(arr: List) -> List:
        n = len(arr)
        if not arr or n <= 1:
            return arr
        Sorting.type_checking(arr)
        for i in range(n):
            j = i
            while j > 0 and arr[j - 1] > arr[j]:
                arr[j - 1], arr[j] = arr[j], arr[j - 1]
                j -= 1
        return arr
    
    @staticmethod
    def selection_sort(arr: List) -> List:
        n = len(arr)
        if not arr or n <= 1:
            return arr
        Sorting.type_checking(arr)
        for i in range(n):
            min = i
            for j in range(i + 1, n):
                if arr[j] < arr[min]:
                    min = j
            arr[min], arr[i] = arr[i], arr[min]
        return arr
    
    @staticmethod
    def merge_sort(arr: List) -> List:
        n = len(arr)
        if not arr or n <= 1:
            return arr
        Sorting.type_checking(arr)
        return Sorting._merge_sort(arr)

    @staticmethod
    def _merge_sort(arr: List) -> List:
        n = len(arr)
        if n <= 1:
            return arr
        mid_idx = n // 2
        left_arr = arr[0 : mid_idx]
        right_arr = arr[mid_idx :]
        left_arr = Sorting._merge_sort(left_arr)
        right_arr = Sorting._merge_sort(right_arr)

        left_idx, right_idx, res_idx = 0, 0, 0
        res_arr = [0] * n
        while left_idx < len(left_arr) and right_idx < len(right_arr):
            if left_arr[left_idx] <= right_arr[right_idx]:
                res_arr[res_idx] = left_arr[left_idx]
                left_idx += 1
            else:
                res_arr[res_idx] = right_arr[right_idx]
                right_idx += 1
            res_idx += 1
        
        while left_idx < len(left_arr):
            res_arr[res_idx] = left_arr[left_idx]
            res_idx += 1
            left_idx += 1

        while right_idx < len(right_arr):
            res_arr[res_idx] = right_arr[right_idx]
            res_idx += 1
            right_idx += 1
        
        return res_arr
    
    @staticmethod
    def quick_sort(arr: List) -> List:
        n = len(arr)
        if not arr or n <= 1:
            return arr
        Sorting.type_checking(arr)
        Sorting._quick_sort(arr, 0, n - 1)
        return arr

    @staticmethod
    def _quick_sort(arr, left, right):
        if left >= right:
            return
        pivot_idx = random.randint(left, right)
        pivot = arr[pivot_idx]
        arr[left], arr[pivot_idx] = arr[pivot_idx], arr[left]
        i, j = left + 1, right
        while i <= j:
            while i <= j and arr[i] <= pivot:
                i += 1
            while i <= j and arr[j] >= pivot:
                j -= 1
            if i <= j:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
                j -= 1
        arr[pivot_idx], arr[j] = arr[j], arr[pivot_idx]
        Sorting._quick_sort(arr, left, j)
        Sorting._quick_sort(arr, j + 1, right)

    @staticmethod
    def heap_sort(arr: List) -> List:
        n = len(arr)
        if not arr or n <= 1:
            return arr
        Sorting.type_checking(arr)
        pq = []
        for i in range(len(arr)):
            heapq.heappush(pq, arr[i])
        for i in range(len(arr)):
            arr[i] = heapq.heappop(pq)
        return arr
    
    @staticmethod
    def radix_sort(arr: List[int]) -> List[int]:
        n = len(arr)
        if not arr or n <= 1:
            return arr
        Sorting.type_checking(arr)
        buckets = [deque() for _ in range(19)]
        n_itr = 0
        max_num = max(arr)
        while max_num != 0:
            n_itr += 1
            max_num //= 10
        divisor = 1
        for _ in range(n_itr, 0, -1):
            for i in range(len(arr)):
                digit = (arr[i] // divisor) % 10
                buckets[digit + 9].append(arr[i])
            idx = 0
            for i in range(19):
                while len(buckets[i]) != 0:
                    arr[idx] = buckets[i].popleft()
                    idx += 1
            divisor *= 10
        return arr
