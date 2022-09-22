import torch
from sklearn.metrics import confusion_matrix


class Oracle(torch.nn.Module):

    def __init__(self, generation_model):
        super(Oracle, self).__init__()

        # the data generation model
        self.generation_model = generation_model

        # number of points for estimating confusion matrix
        num_of_points = 1000000

        # generate large amount of synthetic data
        x, y = self.generation_model.create_data_set(num_of_points)

        # pass it through the oracle
        model_predictions = self.generation_model.compute_classes_probability(x)

        # get the oracle predictions
        predicted_y = torch.argmax(model_predictions, dim=1)

        # calculate confusion matrix
        confusion_mat = torch.from_numpy(confusion_matrix(y, predicted_y))

        # turn confusion matrix into probabilities
        self.confusion_mat = confusion_mat / torch.unsqueeze(torch.sum(confusion_mat, dim=1), dim=1)

        # get histogram of labels
        hist = torch.bincount(y)

        # get frequency of each label
        self.labels_frequency = hist/num_of_points

    # get the oracle predictions
    def forward(self, x):
        x = self.generation_model.compute_classes_probability(x)
        return x


class UniformOracle(torch.nn.Module):

    def __init__(self, generation_model, noise_probability=0.1):
        super(UniformOracle, self).__init__()

        # the data generation model
        self.generation_model = generation_model

        # portion of data corrupted by label noise
        self.noise_probability = noise_probability

    def forward(self, x):
        # get estimated conditional probabilities of the clean data oracle
        x = self.generation_model.compute_classes_probability(x)

        # get number of classes
        num_of_classes = x.size()[1]

        # get oracle predictions under the uniform noise
        x = ((1 - self.noise_probability - self.noise_probability / (num_of_classes - 1)) * x) + self.noise_probability / (num_of_classes - 1)
        return x


class RareToCommonOracle(torch.nn.Module):

    def __init__(self, generation_model, labels_frequency, noise_probability=0.1):
        super(RareToCommonOracle, self).__init__()

        # the data generation model
        self.generation_model = generation_model

        # orders the classes by frequency
        frequency_order = torch.argsort(labels_frequency)

        # check that there are enough probability to change
        assert torch.sum(labels_frequency[frequency_order[0:-1]]) >= noise_probability, 'Not enough labels that are not from the most common class'

        # get number of classes
        num_of_classes = len(labels_frequency)

        # check which label is the most common one
        self.most_common_label = frequency_order[-1]

        # initiate how much to take from each label
        self.portion_from_each_label = torch.zeros(num_of_classes)

        # each iteration change the least common label to the most common one until the desired probability is reached
        left_probability = noise_probability
        for i in range(num_of_classes):

            # check which label is currently the least common one
            least_common_label = frequency_order[i]

            # check the probability to belong to this class
            class_probability = labels_frequency[least_common_label]

            # check if this class probability is enough or more classes are needed
            if class_probability < left_probability:
                # change all the labels from this class
                self.portion_from_each_label[least_common_label] = 1

                # update the amount of probability left
                left_probability = left_probability - class_probability

            # take only a portion of this class to reach the desired noise level
            else:
                self.portion_from_each_label[least_common_label] = left_probability/class_probability
                break

    def forward(self, x):
        # get estimated conditional probabilities of the clean data oracle
        x = self.generation_model.compute_classes_probability(x)

        # get oracle predictions under this noise
        x[:, self.most_common_label] = x[:, self.most_common_label] + torch.sum(self.portion_from_each_label * x, dim=1)
        x = (1 - self.portion_from_each_label) * x

        return x


class CommonMistakeOracle(torch.nn.Module):

    def __init__(self, generation_model, labels_frequency, confusion_matrix, noise_probability=0.1):
        super(CommonMistakeOracle, self).__init__()

        # the data generation model
        self.generation_model = generation_model

        # get number of classes
        self.num_of_classes = confusion_matrix.size()[0]

        # create a local copy
        confusion_mat = torch.clone(confusion_matrix)

        # initiate how much to take from each label
        self.portion_from_each_label = torch.zeros((self.num_of_classes, self.num_of_classes))

        # zero diagonal elements in confusion matrix
        confusion_mat[torch.arange(self.num_of_classes), torch.arange(self.num_of_classes)] = 0

        # each iteration change the most common mistake until there aren't anymore labels to change
        left_probability = noise_probability
        for i in range(self.num_of_classes):
            # check that there are enough labels to change
            assert i != (self.num_of_classes - 1), 'Not enough labels to change'

            # get indices of the most common mistake
            common_mistake = (confusion_mat == torch.max(confusion_mat)).nonzero()
            source_label = common_mistake[0, 0]
            target_label = common_mistake[0, 1]

            # check the probability to belong to this source class
            class_probability = labels_frequency[source_label]

            # make sure this label won't be the target label in future iterations
            confusion_mat[:, source_label] = 0

            # check if this class probability is enough or more classes are needed
            if class_probability < left_probability:
                # change all the labels from this class
                self.portion_from_each_label[target_label, source_label] = 1

                # update the amount of probability left
                left_probability = left_probability - class_probability

            # take only a portion of this class to reach the desired noise level
            else:
                self.portion_from_each_label[target_label, source_label] = left_probability/class_probability
                break

    def forward(self, x):
        # get estimated conditional probabilities of the clean data oracle
        x = self.generation_model.compute_classes_probability(x)

        # get oracle predictions under this noise
        for i in range(self.num_of_classes):
            x[:, i] = x[:, i] + torch.sum(self.portion_from_each_label[i, :] * x, dim=1)
            x = (1 - self.portion_from_each_label[i, :]) * x

        return x


class ConfusionMatrixOracle(torch.nn.Module):

    def __init__(self, generation_model, labels_frequency, confusion_matrix, noise_probability=0.1):
        super(ConfusionMatrixOracle, self).__init__()

        # the data generation model
        self.generation_model = generation_model

        # get number of classes
        num_of_classes = confusion_matrix.size()[0]

        # calculate the true probability to make a mistake
        mistake_probability = torch.sum((1-confusion_matrix[torch.arange(num_of_classes), torch.arange(num_of_classes)])*labels_frequency)

        # create the new corruption matrix
        corruption_matrix = torch.clone(confusion_matrix)
        corruption_matrix = corruption_matrix * (noise_probability/mistake_probability)
        corruption_matrix[torch.arange(num_of_classes), torch.arange(num_of_classes)] = 1-(noise_probability/mistake_probability)*(1-confusion_matrix[torch.arange(num_of_classes), torch.arange(num_of_classes)])
        self.corruption_matrix = corruption_matrix

    def forward(self, x):
        # get estimated conditional probabilities of the clean data oracle
        x = self.generation_model.compute_classes_probability(x)

        # get oracle predictions under this noise
        x = x @ (self.corruption_matrix)
        return x

class TwoLayerNet(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        # initializing the parent object
        super(TwoLayerNet, self).__init__()
        # define the first layer (hidden)
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        # define the second layer (output)
        self.linear2 = torch.nn.Linear(hidden_dim, output_dim)
        # define the activation function
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
