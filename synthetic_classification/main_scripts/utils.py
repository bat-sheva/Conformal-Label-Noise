import torch
import numpy as np
import gc
from torch.nn.functional import softmax
from numpy.random import default_rng
from scipy.stats.mstats import mquantiles
import time


def get_scores(model, x, scores_list, device='cpu', gpu_capacity=1024):
    # get number of points
    n = x.size()[0]

    # get the model predictions for each class
    model_predictions = get_model_logits(model, x, device=device, gpu_capacity=gpu_capacity).to(torch.device('cpu'))

    # get number of classes
    num_of_classes = model_predictions.size()[1]

    # transform the output into probabilities vectors
    if torch.sum(model_predictions[0, :]) != 1:
        model_predictions = softmax(model_predictions, dim=1).numpy()
    else:
        model_predictions = model_predictions.numpy()

    # initiate random uniform variables for APS score
    rng = default_rng()
    uniform_variables = rng.uniform(size=n, low=0.0, high=1.0)

    # create container for the scores
    scores = np.zeros((len(scores_list), n, num_of_classes))

    # run over all scores functions and compute scores
    for p, score_func in enumerate(scores_list):
        scores[p, :, :] = score_func(model_predictions, np.arange(num_of_classes), uniform_variables,
                                     all_combinations=True)

    return scores


def calibration(scores, alpha=0.1):
    # size of the calibration set
    n_calib = scores.shape[1]

    # get number of scores
    num_of_scores = scores.shape[0]

    # create container for the calibration thresholds
    thresholds = np.zeros(num_of_scores)

    # Compute thresholds
    level_adjusted = (1.0 - alpha) * (1.0 + 1.0 / float(n_calib))
    for p in range(num_of_scores):
        thresholds[p] = mquantiles(scores[p, :], prob=level_adjusted)

    return thresholds


def prediction(scores, thresholds):
    # get number of points
    n = scores.shape[1]

    # get number of scores
    num_of_scores = scores.shape[0]

    # Generate prediction sets using the thresholds from the calibration
    predicted_sets = []
    for p in range(num_of_scores):
        predicted_sets.append([np.where(scores[p, i, :] <= thresholds[p])[0] for i in range(n)])

    # return predictions sets
    return predicted_sets


def evaluate_prediction_sets(sets, y):
    # get numbers of points
    n = len(y)

    # turn labels to np array
    y = np.array(y)

    # Marginal coverage
    marginal_coverage = np.mean([y[i] in sets[i] for i in range(n)])

    # Average set size
    average_set_size = np.mean([len(sets[i]) for i in range(n)])

    set_sizes = [len(sets[i]) for i in range(n)]

    return marginal_coverage, average_set_size


def add_label_noise_uniform(labels, probability=0.1, num_of_classes=10):
    # get number of labels
    num_of_labels = labels.size()[0]

    # get number of labels to change
    num_of_labels_to_change = int(probability * num_of_labels)

    # choose indexes to change
    weights = torch.ones(num_of_labels) / num_of_labels
    indexes_to_change = torch.multinomial(weights, num_of_labels_to_change, replacement=False)

    # create a matrix where element (i,j) represents the probability to switch from label i to label j
    switching_probability_mat = (1 / (num_of_classes - 1)) * torch.ones((num_of_classes, num_of_classes))

    # set the probability to not change the label to zero
    switching_probability_mat[torch.arange(num_of_classes), torch.arange(num_of_classes)] = 0

    # container for the output noisy labels
    noisy_labels = torch.clone(labels)

    # sample the noisy labels using the probabilities matrix
    noisy_labels[indexes_to_change] = torch.tensor(
        [torch.multinomial(switching_probability_mat[labels[indexes_to_change[i]], :], 1) for i in
         range(num_of_labels_to_change)], dtype=int)
    return noisy_labels


def add_label_noise_wrong_to_right(model, points, labels, probability=0.1, device='cpu', gpu_capacity=1024):
    # get number of points
    n = labels.size()[0]

    # get the model predictions for each class
    model_predictions = get_model_logits(model, points, device=device, gpu_capacity=gpu_capacity)

    # calculate the most likely label for each point according to the model
    most_likely_y = torch.argmax(model_predictions, dim=1)

    # check which points has a label different then the most likely one, the model is wrong
    potential_indexes = torch.where(labels != most_likely_y)[0]

    # make sure the desired portion of the labels are being changed (if possible)
    assert len(potential_indexes) >= probability * n, 'Not enough wrong labels'
    probability = probability / (len(potential_indexes) / n)

    # container for the output noisy labels
    noisy_labels = torch.clone(labels)

    # get number of labels to change from the potential ones
    num_of_labels_to_change = int(probability * len(potential_indexes))

    # choose indexes to change from the potential indexes
    weights = torch.ones(len(potential_indexes)) / len(potential_indexes)
    indexes_to_change = torch.multinomial(weights, num_of_labels_to_change, replacement=False)

    # change labels of chosen indexes to the most likely label (the model prediction, makes the model thing he was right)
    noisy_labels[potential_indexes[indexes_to_change]] = most_likely_y[potential_indexes[indexes_to_change]]

    return noisy_labels


def add_label_noise_next_most_likely(model, points, labels, probability=0.1, device='cpu', gpu_capacity=1024):
    # get number of points
    n = labels.size()[0]

    # get the model predictions for each class
    model_predictions = get_model_logits(model, points, device=device, gpu_capacity=gpu_capacity)

    # calculate the most likely label for each point according to the model
    most_likely_y = torch.argmax(model_predictions, dim=1)

    # get number of classes
    num_of_classes = model_predictions.size()[1]

    # calculate the second most likely label for each point according to the model
    second_most_likely_y = torch.kthvalue(model_predictions, k=num_of_classes - 1, dim=1)[1]

    # potential new labels
    new_labels = most_likely_y
    new_labels[torch.where(most_likely_y == labels)[0]] = second_most_likely_y[torch.where(most_likely_y == labels)[0]]

    # container for the output noisy labels
    noisy_labels = torch.clone(labels)

    # replace label with probability p to the most likely label other then the current label
    u = torch.rand(n)
    indexes_to_change = torch.where(u <= probability)[0]
    noisy_labels[indexes_to_change] = new_labels[indexes_to_change]

    return noisy_labels


def add_label_noise_rare_to_common(labels, labels_frequency, probability=0.1):
    # get number of points
    n = labels.size()[0]

    # counter for number of changed labels so far
    num_of_changed_labels = 0

    # container for the output noisy labels
    noisy_labels = torch.clone(labels)

    # get number of classes
    num_of_classes = len(labels_frequency)

    # orders the classes by frequency
    frequency_order = torch.argsort(labels_frequency)

    # check which label is the most common one
    most_common_label = frequency_order[-1]

    # each iteration change the least common label to the most common one until the desired portion is changed
    for i in range(num_of_classes):

        # check that there are enough labels to change
        assert i != (num_of_classes - 1), 'Not enough labels that are not from the most common class'

        # check which label is currently the least common one
        least_common_label = frequency_order[i]

        # check which points comes from the rarest class
        potential_indexes = torch.where(labels == least_common_label)[0]

        # check if you need to change all the labels from this class or only a portion
        if len(potential_indexes) < int(probability * n) - num_of_changed_labels:
            # change all the labels from this class
            noisy_labels[potential_indexes] = most_common_label * torch.ones(len(potential_indexes), dtype=int)

            # update the amount of changed labels thus far
            num_of_changed_labels = num_of_changed_labels + len(potential_indexes)

        # change only the additional needed number of labels to reach the desired noise portion
        else:
            # check how many more labels needs to be changed
            num_of_labels_to_change = int(probability * n) - num_of_changed_labels

            # choose indexes to change from the potential indexes
            weights = torch.ones(len(potential_indexes)) / len(potential_indexes)
            indexes_to_change = torch.multinomial(weights, num_of_labels_to_change, replacement=False)

            # change labels of chosen indexes to the most common class
            noisy_labels[potential_indexes[indexes_to_change]] = most_common_label * torch.ones(len(indexes_to_change),
                                                                                                dtype=int)
            break

    return noisy_labels


def add_label_noise_worst(scores, labels, probability=0.1, alpha=0.1, fast=False):
    # turn scores into tensor
    scores = torch.from_numpy(scores)

    # get number of points
    n = labels.size()[0]

    # get the lowest scores (best) a label got for each point
    lowest_labels_scores = torch.min(scores, dim=1)[0]

    # Compute the desired quntile level
    level_adjusted = (1.0 - alpha) * (1.0 + 1.0 / float(n))

    if not fast:

        current_labels = torch.clone(labels)

        # each iteration check which label to change until reaching the desired portion
        for j in range(int(n * probability)):
            # get scores of the current labels
            current_scores = scores[torch.arange(n), current_labels]

            # get the 1-alpha quantile of the scores
            threshold = torch.quantile(current_scores, q=level_adjusted)

            # switch only labels with scores above the threshold
            potential_indexes = torch.where(current_scores >= threshold)[0]

            # choose the score above the threshold that can be switched to a label with the smallest possible score
            index_to_switch = potential_indexes[torch.argmin(lowest_labels_scores[potential_indexes])]

            # switch the label
            current_labels[index_to_switch] = torch.argmin(scores[index_to_switch])
        noisy_labels = current_labels

    else:
        # get the scores of the ground truth labels
        true_labels_scores = scores[torch.arange(n), labels]

        # get the 1-alpha quantile of the scores
        threshold = torch.quantile(true_labels_scores, q=level_adjusted)

        # switch only labels with scores above the threshold
        potential_indexes = torch.where(true_labels_scores > threshold)[0]

        # check that there ar enough labels to change
        assert len(potential_indexes) >= probability * n, 'Not enough labels above the threshold'

        # choose the scores above the threshold that can be switched to a label with the smallest possible score
        indexes_to_switch = potential_indexes[
            torch.topk(lowest_labels_scores[potential_indexes], int(n * probability), largest=False)[1]]

        # container for the output noisy labels
        noisy_labels = torch.clone(labels)

        # switch those labels
        noisy_labels[indexes_to_switch] = torch.argmin(scores[indexes_to_switch], dim=1)

    return noisy_labels


def add_label_noise_confusion_matrix(labels, confusion_matrix, probability=0.1):
    # get number of points
    n = labels.size()[0]

    # get number of classes
    num_of_classes = confusion_matrix.size()[0]

    # get number of labels to change
    num_of_labels_to_change = int(probability * n)

    # create local copy
    confusion_mat = torch.clone(confusion_matrix)

    # choose indexes to change according to the probability to make an error in each class
    weights = torch.tensor([(1 - confusion_mat[labels[i], labels[i]]) for i in range(n)])
    weights = weights / torch.sum(weights)
    indexes_to_change = torch.multinomial(weights, num_of_labels_to_change, replacement=False)

    # zero the diagonal elements in the confusion matrix
    confusion_mat[torch.arange(num_of_classes), torch.arange(num_of_classes)] = torch.zeros(num_of_classes)

    # turn confusion matrix into probabilities again
    confusion_mat = confusion_mat / torch.unsqueeze(torch.sum(confusion_mat, dim=1), dim=1)

    # container for the output noisy labels
    noisy_labels = torch.clone(labels)

    # sample the noisy labels using the confusion matrix probabilities
    noisy_labels[indexes_to_change] = torch.tensor(
        [torch.multinomial(confusion_mat[labels[indexes_to_change[i]], :], 1) for i in range(num_of_labels_to_change)],
        dtype=int)

    return noisy_labels


def add_label_noise_common_mistakes(labels, confusion_matrix, probability=0.1):
    # get number of points
    n = labels.size()[0]

    # get number of classes
    num_of_classes = confusion_matrix.size()[0]

    # create a local copy
    confusion_mat = torch.clone(confusion_matrix)

    # counter for number of changed labels so far
    num_of_changed_labels = 0

    # container for the output noisy labels
    noisy_labels = torch.clone(labels)

    # zero diagonal elements in confusion matrix
    confusion_mat[torch.arange(num_of_classes), torch.arange(num_of_classes)] = 0

    # each iteration change the most common mistake until there aren't anymore labels to change
    for i in range(num_of_classes):
        # check that there are enough labels to change
        assert i != (num_of_classes - 1), 'Not enough labels to change'

        # get indices of the most common mistake
        common_mistake = (confusion_mat == torch.max(confusion_mat)).nonzero()
        source_label = common_mistake[0, 0]
        target_label = common_mistake[0, 1]

        # make sure this label won't be the target label in future iterations
        confusion_mat[:, source_label] = 0

        # check which points comes from the current class
        potential_indexes = torch.where(labels == source_label)[0]

        # check if you need to change all the labels from this class or only a portion
        if len(potential_indexes) < int(probability * n) - num_of_changed_labels:
            # change all the labels from this class
            noisy_labels[potential_indexes] = target_label * torch.ones(len(potential_indexes), dtype=int)

            # update the amount of changed labels thus far
            num_of_changed_labels = num_of_changed_labels + len(potential_indexes)

        # change only the additional needed number of labels to reach the desired noise portion
        else:
            # check how many more labels needs to be changed
            num_of_labels_to_change = int(probability * n) - num_of_changed_labels

            # choose indexes to change from the potential indexes
            weights = torch.ones(len(potential_indexes)) / len(potential_indexes)
            indexes_to_change = torch.multinomial(weights, num_of_labels_to_change, replacement=False)

            # change labels of chosen indexes
            noisy_labels[potential_indexes[indexes_to_change]] = target_label * torch.ones(len(indexes_to_change),
                                                                                           dtype=int)
            break

    return noisy_labels


def get_model_logits(model, x, device='cpu', gpu_capacity=1024):
    # get number of points
    n = x.size()[0]

    # maximum batch size according to gpu capacity
    batch_size = gpu_capacity

    # calculate number of batches
    if n % batch_size != 0:
        num_of_batches = (n // batch_size) + 1
    else:
        num_of_batches = (n // batch_size)

    # get model predictions in batches
    for j in range(num_of_batches):
        # get inputs of batch
        inputs = x[(j * batch_size):((j + 1) * batch_size)].to(device)

        # get classifier predictions
        model.eval()
        with torch.no_grad():
            batch_outputs = model(inputs)

        if j == 0:
            # get number of classes
            num_of_classes = batch_outputs.size()[1]

            # create container for classifier outputs
            outputs = torch.zeros((n, num_of_classes))

        # store batch
        outputs[(j * batch_size):((j + 1) * batch_size), :] = batch_outputs

        del batch_outputs
        gc.collect()

    return outputs


def train_loop(model, x_train, y_train, device="cpu", gpu_capacity=1024):
    # hyper-parameters:
    num_epochs = 10
    learning_rate = 0.001
    batch_size = 32

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # get number of points
    num_of_points = x_train.size()[0]

    # maximum batch size according to gpu capacity
    if batch_size > gpu_capacity:
        batch_size = gpu_capacity

    # calculate number of batches
    if num_of_points % batch_size != 0:
        num_of_batches = (num_of_points // batch_size) + 1
    else:
        num_of_batches = (num_of_points // batch_size)

    # Train the model
    model.train()
    start_time = time.time()
    for epoch in range(num_epochs):
        for j in range(num_of_batches):
            # get inputs and labels of batch
            inputs = x_train[(j * batch_size):((j + 1) * batch_size)].to(device)
            labels = y_train[(j * batch_size):((j + 1) * batch_size)].to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (j + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Time: {:.4f} secs'
                      .format(epoch + 1, num_epochs, j + 1, num_of_batches, loss.item(), time.time() - start_time
                              ))


def calculate_model_accuracy(model, x_test, y_test, device="cpu", gpu_capacity=1024):
    # get number of points
    n = x_test.size()[0]

    # get the model predictions for each class
    model_predictions = get_model_logits(model, x_test, device=device, gpu_capacity=gpu_capacity)

    # get predicted labels
    predicted_labels = torch.argmax(model_predictions, dim=1)

    # get correct predictions
    correct_predictions = predicted_labels[torch.where(predicted_labels == y_test)[0]]

    # get model accuracy
    accuracy = correct_predictions.size()[0] / n

    return accuracy
