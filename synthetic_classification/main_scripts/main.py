# general imports
import argparse
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import sys
from sklearn.model_selection import train_test_split
import plotnine as gg
import os

# My imports
sys.path.insert(0, './')
import main_scripts.Score_Functions as scores
from main_scripts.Data_Generation_Model import GenerationModel1
from main_scripts.Models import *
import main_scripts.utils as ut

# parameters
parser = argparse.ArgumentParser(description='Experiments')
parser.add_argument('-a', '--alpha', default=0.1, type=float, help='Desired nominal marginal coverage')
parser.add_argument('-s', '--splits', default=100, type=int, help='Number of experiments to estimate coverage')
parser.add_argument('--n_test', default=10000, type=int, help='Number of test+calibration points')
parser.add_argument('--n_train', default=50000, type=int, help='Number of training points')
parser.add_argument('--n_classes', default=10, type=int, help='Number of classes for generated data')
parser.add_argument('-d', '--data_dim', default=100, type=int, help='Dimension of data')
parser.add_argument('--noise_probability', default=0.05, type=float, help='Desired portion of the calibration points to add label noise')
parser.add_argument('--batch_size', default=1024*64, type=int, help='Maximum number of points in batch gpu is capable of')
parser.add_argument('--noise_type', default="None", type=str, help='Which type of label noise to add: None/Uniform/Confusion_Matrix/Rare_to_Common/Common_Mistake/Wrong_to_Right/Adversarial_HPS/Adversarial_APS')

args = parser.parse_args()

# initiate parameters
alpha = args.alpha  # desired nominal marginal coverage
n_experiments = args.splits  # number of experiments to estimate coverage
n_test = args.n_test
n_train = args.n_train
num_of_classes = args.n_classes
data_dim = args.data_dim
gpu_capacity = args.batch_size
noise_probability = args.noise_probability
calibration_scores = ['HPS', 'APS']  # score function to check 'HPS', 'APS', 'RAPS'
Noise_type = args.noise_type
is_oracle = True
if Noise_type == "Wrong_to_Right" or Noise_type == "Adversarial_HPS" or Noise_type == "Adversarial_APS":
    is_oracle = False

# Validate parameters
assert 0 <= alpha <= 1, 'Nominal level must be between 0 to 1'

# translate desired scores to their functions and put in a list
scores_list = []
for score in calibration_scores:
    if score == 'HPS':
        scores_list.append(scores.class_probability_score)
    elif score == 'APS':
        scores_list.append(scores.generalized_inverse_quantile_score)
    elif score == 'RAPS':
        scores_list.append(scores.rank_regularized_score)
    else:
        print("Undefined score function")
        exit(1)

# set random seed
seed = 400
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Create Synthetic data
generation_model = GenerationModel1(num_of_classes=num_of_classes, dimension=data_dim, magnitude=1)
x_train, y_train = generation_model.create_data_set(n_train)
x_test, y_test = generation_model.create_data_set(n_test)

# create indices for the test points
indices = torch.arange(n_test)

# create the oracle model for this data
oracle_model = Oracle(generation_model)

# get the confusion matrix of this data
confusion_mat = oracle_model.confusion_mat

# get labels frequency
labels_frequency = oracle_model.labels_frequency

# Initiate Oracle model
if Noise_type == "Uniform":
    oracle_model = UniformOracle(generation_model,noise_probability=noise_probability)
    y_train = ut.add_label_noise_uniform(y_train, probability=noise_probability, num_of_classes=num_of_classes)
elif Noise_type == "Rare_to_Common":
    oracle_model = RareToCommonOracle(generation_model, labels_frequency, noise_probability=noise_probability)
    y_train = ut.add_label_noise_rare_to_common(y_train, labels_frequency, probability=noise_probability)
elif Noise_type == "Common_Mistake":
    oracle_model = CommonMistakeOracle(generation_model, labels_frequency, confusion_mat, noise_probability=noise_probability)
    y_train = ut.add_label_noise_common_mistakes(y_train, confusion_mat, probability=noise_probability)
elif Noise_type == "Confusion_Matrix":
    oracle_model = ConfusionMatrixOracle(generation_model, labels_frequency, confusion_mat, noise_probability=noise_probability)
    y_train = ut.add_label_noise_confusion_matrix(y_train, confusion_mat, probability=noise_probability)
else:
    oracle_model = Oracle(generation_model)

# initiate neural network model
nn_model = TwoLayerNet(input_dim=data_dim, hidden_dim=256, output_dim=num_of_classes)

# automatically choose device: use gpu 0 if it is available o.w. use the cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# print the chosen device
print("device: ", device)

# send models to device
oracle_model.to(device)
nn_model.to(device)

# train nn model
ut.train_loop(nn_model, x_train, y_train, device=device, gpu_capacity=gpu_capacity)

# put models in evaluation mode
oracle_model.eval()
nn_model.eval()

# check models accuracy
oracle_accuracy = ut.calculate_model_accuracy(oracle_model, x_test, y_test, device="cpu", gpu_capacity=gpu_capacity)
nn_accuracy = ut.calculate_model_accuracy(nn_model, x_test, y_test, device=device, gpu_capacity=gpu_capacity)
print("\nOracle accuracy: "+str(oracle_accuracy)+"\n")
print("\nNN accuracy: "+str(nn_accuracy)+"\n")

# calculate non-conformity scores for all points
print("Calculating non-conformity scores:\n")
oracle_non_conformity_scores = ut.get_scores(oracle_model, x_test, scores_list, device="cpu", gpu_capacity=gpu_capacity)
nn_non_conformity_scores = ut.get_scores(nn_model, x_test, scores_list, device=device, gpu_capacity=gpu_capacity)

# create dataframe for storing results
results = pd.DataFrame()

# run for n_experiments data splittings
print("\nRunning experiments for "+str(n_experiments)+" random splits:\n")
for experiment in tqdm(range(n_experiments)):
    # Split test data into calibration and test
    idx_calib, idx_test = train_test_split(indices, test_size=0.5)

    # get the labels of calibration points
    calibration_labels = y_test[idx_calib]

    if Noise_type == "Uniform":
        calibration_labels = ut.add_label_noise_uniform(calibration_labels, probability=noise_probability, num_of_classes=num_of_classes)
    elif Noise_type == "Rare_to_Common":
        calibration_labels = ut.add_label_noise_rare_to_common(calibration_labels, labels_frequency, probability=noise_probability)
    elif Noise_type == "Common_Mistake":
        calibration_labels = ut.add_label_noise_common_mistakes(calibration_labels, confusion_mat, probability=noise_probability)
    elif Noise_type == "Confusion_Matrix":
        calibration_labels = ut.add_label_noise_confusion_matrix(calibration_labels, confusion_mat, probability=noise_probability)
    elif Noise_type == "Wrong_to_Right":
        #oracle_calibration_labels = ut.add_label_noise_wrong_to_right(oracle_model, x_test[idx_calib], calibration_labels, probability=noise_probability, device="cpu", gpu_capacity=gpu_capacity)
        calibration_labels = ut.add_label_noise_wrong_to_right(nn_model, x_test[idx_calib], calibration_labels, probability=noise_probability, device=device, gpu_capacity=gpu_capacity)
    elif Noise_type == "Adversarial_HPS":
        #oracle_calibration_labels = ut.add_label_noise_worst(oracle_non_conformity_scores[0, idx_calib, :], calibration_labels, probability=noise_probability, alpha=alpha)
        calibration_labels = ut.add_label_noise_worst(nn_non_conformity_scores[0, idx_calib, :], calibration_labels, probability=noise_probability, alpha=alpha)
    elif Noise_type == "Adversarial_APS":
        #oracle_calibration_labels = ut.add_label_noise_worst(oracle_non_conformity_scores[1, idx_calib, :], calibration_labels, probability=noise_probability, alpha=alpha)
        calibration_labels = ut.add_label_noise_worst(nn_non_conformity_scores[1, idx_calib, :], calibration_labels, probability=noise_probability, alpha=alpha)

    # oracle_accuracy = ut.calculate_model_accuracy(oracle_model, x_test[idx_calib], calibration_labels, device="cpu", gpu_capacity=gpu_capacity)
    # nn_accuracy = ut.calculate_model_accuracy(nn_model, x_test[idx_calib], calibration_labels, device=device, gpu_capacity=gpu_capacity)
    # print("\nOracle accuracy: " + str(oracle_accuracy) + "\n")
    # print("\nNN accuracy: " + str(nn_accuracy) + "\n")
    # exit(1)

    # calibrate model with the desired scores and get the thresholds
    if is_oracle:
        oracle_thresholds = ut.calibration(oracle_non_conformity_scores[:, idx_calib, calibration_labels], alpha=alpha)
    nn_thresholds = ut.calibration(nn_non_conformity_scores[:, idx_calib, calibration_labels], alpha=alpha)

    # generate prediction sets
    if is_oracle:
        oracle_predicted_sets = ut.prediction(oracle_non_conformity_scores[:, idx_test, :], oracle_thresholds)
    nn_predicted_sets = ut.prediction(nn_non_conformity_scores[:, idx_test, :], nn_thresholds)

    # evaluate prediction sets
    for p in range(len(scores_list)):
        # evaluate prediction sets
        if is_oracle:
            marginal_coverage, average_set_size = ut.evaluate_prediction_sets(oracle_predicted_sets[p], y_test[idx_test])
            experiment_result = pd.DataFrame({'Model': ["Oracle"], 'Noise Type': [str(Noise_type)], 'Score Function': [calibration_scores[p]], 'Marginal Coverage': [marginal_coverage], 'Average Set Size': [average_set_size]})
            results = pd.concat([results, experiment_result])

        marginal_coverage, average_set_size = ut.evaluate_prediction_sets(nn_predicted_sets[p], y_test[idx_test])
        experiment_result = pd.DataFrame({'Model': ["NN"], 'Noise Type': [str(Noise_type)], 'Score Function': [calibration_scores[p]], 'Marginal Coverage': [marginal_coverage], 'Average Set Size': [average_set_size]})
        results = pd.concat([results, experiment_result])


# directory to save results
directory = "./Results/alpha_" + str(alpha) + "/noise_portion_" + str(
        noise_probability) + "/noise_type_" + str(Noise_type)

print("Saving results in: " + str(directory))

# create directory if necessary
if not os.path.exists(directory):
    os.makedirs(directory)

# save results
results.to_csv(directory + "/results.csv")

## plot results
#base_size = 18
#nominal = pd.DataFrame({'name': ['Nominal Level'], 'Coverage': [1 - alpha]})
#
#p = gg.ggplot(results,
#           gg.aes(x="Model", y="Marginal Coverage", color="Score Function")) \
#    + gg.geom_boxplot() \
#    + gg.geom_hline(nominal, gg.aes(yintercept='Coverage', size='name'), linetype="dashed", color="black") \
#    + gg.labs(x="Model", y="Marginal Coverage", title=str(Noise_type)) \
#    + gg.theme_bw(base_size=base_size) \
#    + gg.theme(panel_grid_minor=gg.element_blank(),
#            panel_grid_major=gg.element_line(size=0.2, colour="#d3d3d3"),
#            plot_title=gg.element_text(face="bold"),
#            legend_background=gg.element_rect(fill="white", size=4, colour="white"),
#            text=gg.element_text(size=base_size, face="plain"),
#            legend_title_align='center',
#            legend_position=(-0.3, 0.5),
#            strip_background_y=gg.element_blank(),
#            axis_text_x=gg.element_text(rotation=45, vjust=1, hjust=1),
#            subplots_adjust={'hspace': 0.05},
#            legend_entry_spacing=10,
#            legend_direction='horizontal') \
#    + gg.scale_size_manual(name=" ", values=(1, 1)) \
#    + gg.guides(color=gg.guide_legend(order=1)) \
#    + gg.scale_y_continuous(expand=(0.1, 0, 0.1, 0))
#
#p.save(directory + "/Marginal_Coverage.pdf")
#
#p = gg.ggplot(results,
#           gg.aes(x="Model", y="Average Set Size", color="Score Function")) \
#    + gg.labs(x="Model", y="Average Set Size", title=str(Noise_type)) \
#    + gg.theme_bw(base_size=base_size) \
#    + gg.theme(legend_title_align='center',
#            panel_grid_minor=gg.element_blank(),
#            panel_grid_major=gg.element_line(size=0.2, colour="#d3d3d3"),
#            plot_title=gg.element_text(face="bold"),
#            legend_background=gg.element_rect(fill="white", size=4, colour="white"),
#            text=gg.element_text(size=base_size, face="plain"),
#            legend_position="none",
#            axis_text_x=gg.element_text(rotation=45, vjust=1, hjust=1),
#            legend_direction='horizontal',
#            legend_entry_spacing=10,
#            subplots_adjust={'wspace': 0.4}) \
#    + gg.scale_y_continuous(expand=(0.1, 0, 0.1, 0)) \
#    + gg.geom_boxplot()
#
#p.save(directory + "/Average_Set_Size.pdf")
