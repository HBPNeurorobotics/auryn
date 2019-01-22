from sortedcontainers import SortedList


class OnlineMedianFinder:

    def __init__(self):
        self.currentMedian = 0
        self.bigElements = SortedList()
        self.smallElements = SortedList()
        self.balance = 0

    def add_element(self, newval):
        if self.balance == 0:
            if newval < self.currentMedian:
                self.smallElements.add(newval)
                self.currentMedian = self.smallElements[-1]
                self.balance = -1
            else:
                self.bigElements.add(newval)
                self.currentMedian = self.bigElements[0]
                self.balance = +1
            return
        elif self.balance == +1:
            if newval <= self.currentMedian:
                self.smallElements.add(newval)
            else:
                self.bigElements.add(newval)
                self.smallElements.add(self.bigElements.pop(0))
        elif self.balance == -1:
            if newval > self.currentMedian:
                self.bigElements.add(newval)
            else:
                self.smallElements.add(newval)
                self.bigElements.add(self.smallElements.pop(-1))
        self.currentMedian = self.bigElements[0]
        self.balance = 0

    def remove_element(self, oldval):
        if self.balance == 0:
            if oldval < self.currentMedian:
                self.smallElements.remove(oldval)
                self.currentMedian = self.bigElements[0]
                self.balance = +1
            else:
                self.bigElements.remove(oldval)
                self.currentMedian = self.smallElements[-1]
                self.balance = -1
            return
        elif self.balance == +1:
            if oldval < self.currentMedian:
                self.smallElements.remove(oldval)
                self.smallElements.add(self.bigElements.pop(0))
            else:
                self.bigElements.remove(oldval)
        elif self.balance == -1:
            if oldval >= self.currentMedian:
                self.bigElements.remove(oldval)
                self.bigElements.add(self.smallElements.pop(-1))
            else:
                self.smallElements.remove(oldval)
        self.currentMedian = self.bigElements[0]
        self.balance = 0
