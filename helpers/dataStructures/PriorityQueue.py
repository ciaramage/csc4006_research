import numpy as np
import heapq as hq
import itertools

"""Based off priority queue implementation from:
    https://docs.python.org/3/library/heapq.html
"""


class PriorityQueue():

    def __init__(self, items, priorities):
        self.priorityQueue = []
        self.item_finder = {}
        self.counter = itertools.count()

        for item, priority in zip(items, priorities):
            priorityItem = [priority, next(self.counter), item]
            self.priorityQueue.append(priorityItem)
            self.item_finder[item] = priorityItem
        hq.heapify(self.priorityQueue)


    def add_item(self, item, priority):
        if item in self.item_finder:
            self.remove_item(item)
       
        priorityItem = [priority, next(self.counter), item]
        self.item_finder[item] = priorityItem
        hq.heappush(self.priorityQueue, priorityItem)
    
        
    def remove_item(self, item):
        entry = self.item_finder.pop(item)
        entry[-1] = "REMOVED"
    
    def pop_item(self):
        while self.priorityQueue:
            priority, _, item = hq.heappop(self.priorityQueue)
            if item != "REMOVED":
                del self.item_finder[item]
                return priority, item
        raise KeyError("No elements remain in priority queue")
    
    def smallest_item(self):
        priority, _, item = hq.nsmallest(1, self.priorityQueue)[0]
        return priority, item
    
    def largest_item(self):
        priority, _, item = hq.nlargest(1, self.priorityQueue)[0]
        return priority, item

    def nth_largest_item(self, n):
        # (n+1)th largest item that has not been removed from the priority queue
        item = hq.nlargest(n+1, self.priorityQueue, key=lambda r: r[-1] != 'REMOVED')[-1]
        return item[0], item[2] # item[0], item[2] -> priority, item

        
