import time
from functools import reduce

class Timer:
    def __init__(self, nsteps = 1):
        self.nsteps = nsteps
        self.prevStep = 1
        self.start = time.time()
        self.prev = self.start
        self.laps = []

    def elapsed(self, step = 0, round=True):
        # get eleapsed        
        end = time.time()
        prev = self.prev
        self.prev = end
        lap = end - prev
        
        self.laps.append(lap)
        self.prevStep = step

        output = "{0:.2f}".format(lap) if round else lap
        return output

    def printTime(self, timeDetal):
        return time.strftime("%H:%M:%S",  time.gmtime(timeDetal))
    
    def elpasedTot(self):
        return self.printTime(time.time() - self.start)

    def left(self):
        avLap = reduce((lambda x, y: x + y), self.laps, 0)/self.prevStep
        remainininTime = avLap * (self.nsteps - self.prevStep)
        return self.printTime(remainininTime)