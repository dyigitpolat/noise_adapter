class ClassificationMode:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def mode(self):
        return "classification"
    
class RegressionMode:
    def mode(self):
        return "regression"