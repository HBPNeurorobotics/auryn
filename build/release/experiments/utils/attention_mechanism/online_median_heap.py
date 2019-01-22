import heapq


class OnlineMedianFinder:

    def __init__(self):
        self.currentMedian = 0
        self.bigElements = MinHeap()
        self.smallElements = MaxHeap()
        self.balance = 0

    def add_element(self, newval):
        if self.balance == 0:
            if newval < self.currentMedian:
                self.smallElements.heappush(newval)
                self.currentMedian = self.smallElements[0]
                self.balance = -1
            else:
                self.bigElements.heappush(newval)
                self.currentMedian = self.bigElements[0]
                self.balance = +1
            return
        elif self.balance == +1:
            if newval <= self.currentMedian:
                self.smallElements.heappush(newval)
            else:
                self.bigElements.heappush(newval)
                self.smallElements.heappush(self.bigElements.heappop())
        elif self.balance == -1:
            if newval > self.currentMedian:
                self.bigElements.heappush(newval)
            else:
                self.smallElements.heappush(newval)
                self.bigElements.heappush(self.smallElements.heappop())
        self.currentMedian = self.smallElements[0]
        self.balance = 0

    def remove_element(self, oldval):
        if self.balance == 0:
            if oldval < self.currentMedian:
                self.smallElements.remove(oldval)
                self.currentMedian = self.bigElements[0]
                self.balance = +1
            else:
                self.bigElements.remove(oldval)
                self.currentMedian = self.smallElements[0]
                self.balance = -1
            return
        elif self.balance == +1:
            if oldval < self.currentMedian:
                self.smallElements.remove(oldval)
                self.smallElements.heappush(self.bigElements.heappop())
            else:
                self.bigElements.remove(oldval)
        elif self.balance == -1:
            if oldval >= self.currentMedian:
                self.bigElements.remove(oldval)
                self.bigElements.heappush(self.smallElements.heappop())
            else:
                self.smallElements.remove(oldval)
        self.currentMedian = self.smallElements[0]
        self.balance = 0


class MinHeap(object):
    def __init__(self): self.h = []

    def heappush(self, x): heapq.heappush(self.h, x)

    def heappop(self): return heapq.heappop(self.h)

    def __getitem__(self, i): return self.h[i]

    def __len__(self): return len(self.h)

    def getheap(self): return self.h

    def remove(self, x):
        self.h.remove(x)
        heapq.heapify(self.h)


class MaxHeap(object):
    def __init__(self): self.h = []

    def heappush(self, x): heapq.heappush(self.h, -x)

    def heappop(self): return -heapq.heappop(self.h)

    def __getitem__(self, i): return -self.h[i]

    def __len__(self): return len(self.h)

    def getheap(self): return [-i for i in self.h]

    def remove(self, x):
        self.h.remove(-x)
        heapq.heapify(self.h)
