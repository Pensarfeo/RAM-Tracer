class Banana():
    def __call__(self):
        self.state, self.output = [1,2]
        return self.state, self.output


banana = Banana()
print(banana())
