import torch


class GenerationModel1:
    def __init__(self, num_of_classes=10, dimension=10, magnitude=1):
        # initialize generation model parameters
        self.num_of_classes = num_of_classes
        self.dimension = dimension
        self.magnitude = magnitude

        # Generate model
        self.beta_Z = self.magnitude * torch.randn(self.dimension, self.num_of_classes)

    def sample_x(self, n=50000):
        x = torch.randn(n, self.dimension)
        # factor = 0.2
        # x[0:int(n * factor), 0] = 20
        # x[int(n * factor):, 0] = -8
        return x

    def compute_logits(self, x):
        logits = torch.matmul(x, self.beta_Z)
        return logits

    def compute_classes_probability(self, x):
        logits = self.compute_logits(x)
        temperature = 1 # 10
        # logits[torch.where(x[:, 0] == 1)[0], 2] = 50
        # prob_y = torch.nn.functional.softmax(logits, dim=1)
        tmp = torch.exp(logits/temperature)
        prob_y = tmp / torch.unsqueeze(torch.sum(tmp, dim=1), dim=1)
        return prob_y

    def sample_y(self, x):
        prob_y = self.compute_classes_probability(x)
        num_of_points = x.size()[0]
        y = torch.tensor([torch.multinomial(prob_y[i], 1) for i in range(num_of_points)], dtype=int)
        return y

    def create_data_set(self, n=50000):
        x = self.sample_x(n)
        y = self.sample_y(x)
        return x, y
