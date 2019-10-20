from utils import *
from kmnist import *
from fashionmnist import *
from mnist import *
from cifar10 import *
from stl10 import *
from cifar100 import *
import os
import shutil
import datetime

config = config()

now = str(datetime.datetime.now())[:19]
now = now.replace(":","_")
now = now.replace("-","_")
now = now.replace(" ","_")

src_dir = config.path.data_path
path = config.path.result_path + str(config.statistics.dataset) + "_" + str(config.statistics.type) + "_" + str(now)
os.mkdir(path)
dst_dir = path+"/config.yaml"
shutil.copy(src_dir,dst_dir)



if config.statistics.dataset == "KMNIST":
    if config.statistics.type == "training_size":
        print("START STATS ON TRAINING SIZE WITH MNIST : ", config.statistics.training_size_value)
        res_precision = np.zeros(len(config.statistics.training_size_value))
        res_recall = np.zeros(len(config.statistics.training_size_value))
        res_accuracy = np.zeros(len(config.statistics.training_size_value))
        #res_best_acc_target = np.zeros(len(config.statistics.training_size_value))
        #res_best_acc_shadows = np.zeros(len(config.statistics.training_size_value))
        res_precision_per_class = np.zeros((len(config.statistics.training_size_value), 10))
        res_recall_per_class = np.zeros((len(config.statistics.training_size_value), 10))
        res_accuracy_per_class = np.zeros((len(config.statistics.training_size_value), 10))
        for index_value_to_test, value_to_test in enumerate(config.statistics.training_size_value):
            config.set_subkey("general", "train_target_size", value_to_test)
            precision_general, recall_general, accuracy_general, precision_per_class, recall_per_class, accuracy_per_class = kmnist(config, path, index_value_to_test)
            res_precision[index_value_to_test] = precision_general
            res_recall[index_value_to_test] = recall_general
            res_accuracy[index_value_to_test] = accuracy_general
            #res_best_acc_target[index_value_to_test] = best_acc_target
            #res_best_acc_shadows[index_value_to_test] = best_acc_shadows
            res_precision_per_class[index_value_to_test] = precision_per_class
            res_recall_per_class[index_value_to_test] = recall_per_class
            res_accuracy_per_class[index_value_to_test] = accuracy_per_class
            plot_title = "KMNIST"
            drawPlot(accuracy_per_class, precision_per_class, recall_per_class, plot_title, path)
            print("accuracy_per_class:",accuracy_per_class)
            print("precision_per_class:",precision_per_class)
            print("recall_per_class:",recall_per_class)
            np.savetxt(path + "/res_precision.csv", res_precision)
            np.savetxt(path + "/res_recall.csv", res_recall)
            np.savetxt(path + "/res_accuracy.csv", res_accuracy)
            #np.savetxt(path + "/res_best_acc_target.csv", res_best_acc_target)
            np.savetxt(path + "/res_recall_per_class.csv", res_recall_per_class)
            np.savetxt(path + "/res_precision_per_class.csv", res_precision_per_class)
            np.savetxt(path + "/res_accuracy_per_class.csv", res_accuracy_per_class)
            #np.savetxt(path + "/res_best_acc_shadows.csv", res_best_acc_shadows)
    if config.statistics.type == "number_shadow":
        print("START STATS ON NUMBER SHADOW  WITH MNIST : ", config.statistics.number_shadow_value)
        res_precision = np.zeros(len(config.statistics.number_shadow_value))
        res_recall = np.zeros(len(config.statistics.number_shadow_value))
        res_accuracy = np.zeros(len(config.statistics.number_shadow_value))
        #res_best_acc_target = np.zeros(len(config.statistics.number_shadow_value))
        #res_best_acc_shadows = np.zeros(len(config.statistics.number_shadow_value))
        res_precision_per_class = np.zeros((len(config.statistics.number_shadow_value), 10))
        res_recall_per_class = np.zeros((len(config.statistics.number_shadow_value), 10))
        res_accuracy_per_class = np.zeros((len(config.statistics.number_shadow_value), 10))
        for index_value_to_test, value_to_test in enumerate(config.statistics.number_shadow_value):
            config.set_subkey("general", "number_shadow_model", value_to_test)
            precision_general, recall_general, accuracy_general, precision_per_class, recall_per_class, accuracy_per_class = kmnist(config, path, index_value_to_test)
            res_precision[index_value_to_test] = precision_general
            res_recall[index_value_to_test] = recall_general
            res_accuracy[index_value_to_test] = accuracy_general
            #res_best_acc_target[index_value_to_test] = best_acc_target
            #res_best_acc_shadows[index_value_to_test] = best_acc_shadows
            res_precision_per_class[index_value_to_test] = precision_per_class
            res_recall_per_class[index_value_to_test] = recall_per_class
            res_accuracy_per_class[index_value_to_test] = accuracy_per_class
            plot_title = "KMNIST"
            drawPlot(accuracy_per_class, precision_per_class, recall_per_class, plot_title, path)
            print("accuracy_per_class:", accuracy_per_class)
            print("precision_per_class:", precision_per_class)
            print("recall_per_class:", recall_per_class)
            np.savetxt(path + "/res_precision.csv", res_precision)
            np.savetxt(path + "/res_recall.csv", res_recall)
            np.savetxt(path + "/res_accuracy.csv", res_accuracy)
            #np.savetxt(path + "/res_best_acc_target.csv", res_best_acc_target)
            np.savetxt(path + "/res_recall_per_class.csv", res_recall_per_class)
            np.savetxt(path + "/res_precision_per_class.csv", res_precision_per_class)
            np.savetxt(path + "/res_accuracy_per_class.csv", res_accuracy_per_class)
            #np.savetxt(path + "/res_best_acc_shadows.csv", res_best_acc_shadows)
    if config.statistics.type == "overfitting":
        print("START STATS ON OVERFITTING WITH KMNIST : ", config.statistics.epoch_value)
        res_precision = np.zeros(len(config.statistics.epoch_value))
        res_recall = np.zeros(len(config.statistics.epoch_value))
        res_accuracy = np.zeros(len(config.statistics.epoch_value))
        #res_best_acc_target = np.zeros(len(config.statistics.epoch_value))
        #res_best_acc_shadows = np.zeros(len(config.statistics.epoch_value))
        res_precision_per_class = np.zeros((len(config.statistics.epoch_value), 10))
        res_recall_per_class = np.zeros((len(config.statistics.epoch_value), 10))
        res_accuracy_per_class = np.zeros((len(config.statistics.epoch_value), 10))
        for index_value_to_test, value_to_test in enumerate(config.statistics.epoch_value):
            config.set_subkey("learning", "epochs", value_to_test)
            precision_general, recall_general, accuracy_general, precision_per_class, recall_per_class, accuracy_per_class = kmnist(config, path, index_value_to_test)
            res_precision[index_value_to_test] = precision_general
            res_recall[index_value_to_test] = recall_general
            res_accuracy[index_value_to_test] = accuracy_general
            #res_best_acc_target[index_value_to_test] = best_acc_target
            #res_best_acc_shadows[index_value_to_test] = best_acc_shadows
            res_precision_per_class[index_value_to_test] = precision_per_class
            res_recall_per_class[index_value_to_test] = recall_per_class
            res_accuracy_per_class[index_value_to_test] = accuracy_per_class
            plot_title = "KMNIST"
            drawPlot(accuracy_per_class, precision_per_class, recall_per_class, plot_title, path)
            print("accuracy_per_class:", accuracy_per_class)
            print("precision_per_class:", precision_per_class)
            print("recall_per_class:", recall_per_class)
            np.savetxt(path + "/res_precision.csv", res_precision)
            np.savetxt(path + "/res_recall.csv", res_recall)
            np.savetxt(path + "/res_accuracy.csv", res_accuracy)
            #np.savetxt(path + "/res_best_acc_target.csv", res_best_acc_target)
            np.savetxt(path + "/res_recall_per_class.csv", res_recall_per_class)
            np.savetxt(path + "/res_precision_per_class.csv", res_precision_per_class)
            np.savetxt(path + "/res_accuracy_per_class.csv", res_accuracy_per_class)
            #np.savetxt(path + "/res_best_acc_shadows.csv", res_best_acc_shadows)

if config.statistics.dataset == "MNIST":
    if config.statistics.type == "training_size":
        print("START STATS ON TRAINING SIZE WITH MNIST : ", config.statistics.training_size_value)
        res_precision = np.zeros(len(config.statistics.training_size_value))
        res_recall = np.zeros(len(config.statistics.training_size_value))
        res_accuracy = np.zeros(len(config.statistics.training_size_value))
        #res_best_acc_target = np.zeros(len(config.statistics.training_size_value))
        #res_best_acc_shadows = np.zeros(len(config.statistics.training_size_value))
        res_precision_per_class = np.zeros((len(config.statistics.training_size_value), 10))
        res_recall_per_class = np.zeros((len(config.statistics.training_size_value), 10))
        res_accuracy_per_class = np.zeros((len(config.statistics.training_size_value), 10))
        for index_value_to_test, value_to_test in enumerate(config.statistics.training_size_value):
            config.set_subkey("general", "train_target_size", value_to_test)
            precision_general, recall_general, accuracy_general, precision_per_class, recall_per_class, accuracy_per_class = mnist(config, path, index_value_to_test)
            res_precision[index_value_to_test] = precision_general
            res_recall[index_value_to_test] = recall_general
            res_accuracy[index_value_to_test] = accuracy_general
            #res_best_acc_target[index_value_to_test] = best_acc_target
            #res_best_acc_shadows[index_value_to_test] = best_acc_shadows
            res_precision_per_class[index_value_to_test] = precision_per_class
            res_recall_per_class[index_value_to_test] = recall_per_class
            res_accuracy_per_class[index_value_to_test] = accuracy_per_class
            plot_title = "MNIST"
            drawPlot(accuracy_per_class, precision_per_class, recall_per_class, plot_title, path)
            print("accuracy_per_class:", accuracy_per_class)
            print("precision_per_class:", precision_per_class)
            print("recall_per_class:", recall_per_class)
            np.savetxt(path + "/res_precision.csv", res_precision)
            np.savetxt(path + "/res_recall.csv", res_recall)
            np.savetxt(path + "/res_accuracy.csv", res_accuracy)
            #np.savetxt(path + "/res_best_acc_target.csv", res_best_acc_target)
            np.savetxt(path + "/res_recall_per_class.csv", res_recall_per_class)
            np.savetxt(path + "/res_precision_per_class.csv", res_precision_per_class)
            np.savetxt(path + "/res_accuracy_per_class.csv", res_accuracy_per_class)
            #np.savetxt(path + "/res_best_acc_shadows.csv", res_best_acc_shadows)
    if config.statistics.type == "number_shadow":
        print("START STATS ON NUMBER SHADOW  WITH MNIST : ", config.statistics.number_shadow_value)
        res_precision = np.zeros(len(config.statistics.number_shadow_value))
        res_recall = np.zeros(len(config.statistics.number_shadow_value))
        res_accuracy = np.zeros(len(config.statistics.number_shadow_value))
        #res_best_acc_target = np.zeros(len(config.statistics.number_shadow_value))
        #res_best_acc_shadows = np.zeros(len(config.statistics.number_shadow_value))
        res_precision_per_class = np.zeros((len(config.statistics.number_shadow_value), 10))
        res_recall_per_class = np.zeros((len(config.statistics.number_shadow_value), 10))
        res_accuracy_per_class = np.zeros((len(config.statistics.number_shadow_value), 10))
        for index_value_to_test, value_to_test in enumerate(config.statistics.number_shadow_value):
            config.set_subkey("general", "number_shadow_model", value_to_test)
            precision_general, recall_general, accuracy_general, precision_per_class, recall_per_class, accuracy_per_class = mnist(config, path, index_value_to_test)
            res_precision[index_value_to_test] = precision_general
            res_recall[index_value_to_test] = recall_general
            res_accuracy[index_value_to_test] = accuracy_general
            #res_best_acc_target[index_value_to_test] = best_acc_target
            #res_best_acc_shadows[index_value_to_test] = best_acc_shadows
            res_precision_per_class[index_value_to_test] = precision_per_class
            res_recall_per_class[index_value_to_test] = recall_per_class
            res_accuracy_per_class[index_value_to_test] = accuracy_per_class
            plot_title = "MNIST"
            drawPlot(accuracy_per_class, precision_per_class, recall_per_class, plot_title, path)
            print("accuracy_per_class:", accuracy_per_class)
            print("precision_per_class:", precision_per_class)
            print("recall_per_class:", recall_per_class)
            np.savetxt(path + "/res_precision.csv", res_precision)
            np.savetxt(path + "/res_recall.csv", res_recall)
            np.savetxt(path + "/res_accuracy.csv", res_accuracy)
            #np.savetxt(path + "/res_best_acc_target.csv", res_best_acc_target)
            np.savetxt(path + "/res_recall_per_class.csv", res_recall_per_class)
            np.savetxt(path + "/res_precision_per_class.csv", res_precision_per_class)
            np.savetxt(path + "/res_accuracy_per_class.csv", res_accuracy_per_class)
            #np.savetxt(path + "/res_best_acc_shadows.csv", res_best_acc_shadows)
    if config.statistics.type == "overfitting":
        print("START STATS ON OVERFITTING WITH MNIST : ", config.statistics.epoch_value)
        res_precision = np.zeros(len(config.statistics.epoch_value))
        res_recall = np.zeros(len(config.statistics.epoch_value))
        res_accuracy = np.zeros(len(config.statistics.epoch_value))
        #res_best_acc_target = np.zeros(len(config.statistics.epoch_value))
        #res_best_acc_shadows = np.zeros(len(config.statistics.epoch_value))
        res_precision_per_class = np.zeros((len(config.statistics.epoch_value), 10))
        res_recall_per_class = np.zeros((len(config.statistics.epoch_value), 10))
        res_accuracy_per_class = np.zeros((len(config.statistics.epoch_value), 10))
        for index_value_to_test, value_to_test in enumerate(config.statistics.epoch_value):
            config.set_subkey("learning", "epochs", value_to_test)
            precision_general, recall_general, accuracy_general, precision_per_class, recall_per_class, accuracy_per_class = mnist(config, path, index_value_to_test)
            res_precision[index_value_to_test] = precision_general
            res_recall[index_value_to_test] = recall_general
            res_accuracy[index_value_to_test] = accuracy_general
            #res_best_acc_target[index_value_to_test] = best_acc_target
            #res_best_acc_shadows[index_value_to_test] = best_acc_shadows
            res_precision_per_class[index_value_to_test] = precision_per_class
            res_recall_per_class[index_value_to_test] = recall_per_class
            res_accuracy_per_class[index_value_to_test] = accuracy_per_class
            plot_title = "MNIST"
            drawPlot(accuracy_per_class, precision_per_class, recall_per_class, plot_title, path)
            print("accuracy_per_class:", accuracy_per_class)
            print("precision_per_class:", precision_per_class)
            print("recall_per_class:", recall_per_class)
            np.savetxt(path + "/res_precision.csv", res_precision)
            np.savetxt(path + "/res_recall.csv", res_recall)
            np.savetxt(path + "/res_accuracy.csv", res_accuracy)
            #np.savetxt(path + "/res_best_acc_target.csv", res_best_acc_target)
            np.savetxt(path + "/res_recall_per_class.csv", res_recall_per_class)
            np.savetxt(path + "/res_precision_per_class.csv", res_precision_per_class)
            np.savetxt(path + "/res_accuracy_per_class.csv", res_accuracy_per_class)
            #np.savetxt(path + "/res_best_acc_shadows.csv", res_best_acc_shadows)


if config.statistics.dataset == "FashionMNIST":
    if config.statistics.type == "training_size":
        print("START STATS ON TRAINING SIZE WITH FashionMNIST : ", config.statistics.training_size_value)
        res_precision = np.zeros(len(config.statistics.training_size_value))
        res_recall = np.zeros(len(config.statistics.training_size_value))
        res_accuracy = np.zeros(len(config.statistics.training_size_value))
        #res_best_acc_target = np.zeros(len(config.statistics.training_size_value))
        #res_best_acc_shadows = np.zeros(len(config.statistics.training_size_value))
        res_precision_per_class = np.zeros((len(config.statistics.training_size_value), 10))
        res_recall_per_class = np.zeros((len(config.statistics.training_size_value), 10))
        res_accuracy_per_class = np.zeros((len(config.statistics.training_size_value), 10))
        for index_value_to_test, value_to_test in enumerate(config.statistics.training_size_value):
            config.set_subkey("general", "train_target_size", value_to_test)
            precision_general, recall_general, accuracy_general, precision_per_class, recall_per_class, accuracy_per_class = fashionmnist(config, path, index_value_to_test)
            res_precision[index_value_to_test] = precision_general
            res_recall[index_value_to_test] = recall_general
            res_accuracy[index_value_to_test] = accuracy_general
            #res_best_acc_target[index_value_to_test] = best_acc_target
            #res_best_acc_shadows[index_value_to_test] = best_acc_shadows
            res_precision_per_class[index_value_to_test] = precision_per_class
            res_recall_per_class[index_value_to_test] = recall_per_class
            res_accuracy_per_class[index_value_to_test] = accuracy_per_class
            plot_title = "FashionMNIST"
            drawPlot(accuracy_per_class, precision_per_class, recall_per_class, plot_title, path)
            print("accuracy_per_class:", accuracy_per_class)
            print("precision_per_class:", precision_per_class)
            print("recall_per_class:", recall_per_class)
            np.savetxt(path + "/res_precision.csv", res_precision)
            np.savetxt(path + "/res_recall.csv", res_recall)
            np.savetxt(path + "/res_accuracy.csv", res_accuracy)
            #np.savetxt(path + "/res_best_acc_target.csv", res_best_acc_target)
            np.savetxt(path + "/res_recall_per_class.csv", res_recall_per_class)
            np.savetxt(path + "/res_precision_per_class.csv", res_precision_per_class)
            np.savetxt(path + "/res_accuracy_per_class.csv", res_accuracy_per_class)
            #np.savetxt(path + "/res_best_acc_shadows.csv", res_best_acc_shadows)
    if config.statistics.type == "number_shadow":
        print("START STATS ON NUMBER SHADOW  WITH FashionMNIST : ", config.statistics.number_shadow_value)
        res_precision = np.zeros(len(config.statistics.number_shadow_value))
        res_recall = np.zeros(len(config.statistics.number_shadow_value))
        res_accuracy = np.zeros(len(config.statistics.number_shadow_value))
        #res_best_acc_target = np.zeros(len(config.statistics.number_shadow_value))
        #res_best_acc_shadows = np.zeros(len(config.statistics.number_shadow_value))
        res_precision_per_class = np.zeros((len(config.statistics.number_shadow_value), 10))
        res_recall_per_class = np.zeros((len(config.statistics.number_shadow_value), 10))
        res_accuracy_per_class = np.zeros((len(config.statistics.number_shadow_value), 10))
        for index_value_to_test, value_to_test in enumerate(config.statistics.number_shadow_value):
            config.set_subkey("general", "number_shadow_model", value_to_test)
            precision_general, recall_general, accuracy_general, precision_per_class, recall_per_class, accuracy_per_class = fashionmnist(config, path, index_value_to_test)
            res_precision[index_value_to_test] = precision_general
            res_recall[index_value_to_test] = recall_general
            res_accuracy[index_value_to_test] = accuracy_general
            #res_best_acc_target[index_value_to_test] = best_acc_target
            #res_best_acc_shadows[index_value_to_test] = best_acc_shadows
            res_precision_per_class[index_value_to_test] = precision_per_class
            res_recall_per_class[index_value_to_test] = recall_per_class
            res_accuracy_per_class[index_value_to_test] = accuracy_per_class
            plot_title = "FashionMNIST"
            print("accuracy_per_class:", accuracy_per_class)
            print("precision_per_class:", precision_per_class)
            print("recall_per_class:", recall_per_class)
            drawPlot(accuracy_per_class, precision_per_class, recall_per_class, plot_title, path)
            np.savetxt(path + "/res_precision.csv", res_precision)
            np.savetxt(path + "/res_recall.csv", res_recall)
            np.savetxt(path + "/res_accuracy.csv", res_accuracy)
            #np.savetxt(path + "/res_best_acc_target.csv", res_best_acc_target)
            np.savetxt(path + "/res_recall_per_class.csv", res_recall_per_class)
            np.savetxt(path + "/res_precision_per_class.csv", res_precision_per_class)
            np.savetxt(path + "/res_accuracy_per_class.csv", res_accuracy_per_class)
            #np.savetxt(path + "/res_best_acc_shadows.csv", res_best_acc_shadows)
    if config.statistics.type == "overfitting":
        print("START STATS ON OVERFITTING WITH FashionMNIST : ", config.statistics.epoch_value)
        res_precision = np.zeros(len(config.statistics.epoch_value))
        res_recall = np.zeros(len(config.statistics.epoch_value))
        res_accuracy = np.zeros(len(config.statistics.epoch_value))
        #res_best_acc_target = np.zeros(len(config.statistics.epoch_value))
        #res_best_acc_shadows = np.zeros(len(config.statistics.epoch_value))
        res_precision_per_class = np.zeros((len(config.statistics.epoch_value), 10))
        res_recall_per_class = np.zeros((len(config.statistics.epoch_value), 10))
        res_accuracy_per_class = np.zeros((len(config.statistics.epoch_value), 10))
        for index_value_to_test, value_to_test in enumerate(config.statistics.epoch_value):
            config.set_subkey("learning", "epochs", value_to_test)
            precision_general, recall_general, accuracy_general, precision_per_class, recall_per_class, accuracy_per_class = fashionmnist(config, path, index_value_to_test)
            res_precision[index_value_to_test] = precision_general
            res_recall[index_value_to_test] = recall_general
            res_accuracy[index_value_to_test] = accuracy_general
            #res_best_acc_target[index_value_to_test] = best_acc_target
            #res_best_acc_shadows[index_value_to_test] = best_acc_shadows
            res_precision_per_class[index_value_to_test] = precision_per_class
            res_recall_per_class[index_value_to_test] = recall_per_class
            res_accuracy_per_class[index_value_to_test] = accuracy_per_class
            plot_title = "FashionMNIST"
            drawPlot(accuracy_per_class, precision_per_class, recall_per_class, plot_title, path)
            print("accuracy_per_class:", accuracy_per_class)
            print("precision_per_class:", precision_per_class)
            print("recall_per_class:", recall_per_class)
            np.savetxt(path + "/res_precision.csv", res_precision)
            np.savetxt(path + "/res_recall.csv", res_recall)
            np.savetxt(path + "/res_accuracy.csv", res_accuracy)
            #np.savetxt(path + "/res_best_acc_target.csv", res_best_acc_target)
            np.savetxt(path + "/res_recall_per_class.csv", res_recall_per_class)
            np.savetxt(path + "/res_precision_per_class.csv", res_precision_per_class)
            np.savetxt(path + "/res_accuracy_per_class.csv", res_accuracy_per_class)
            #np.savetxt(path + "/res_best_acc_shadows.csv", res_best_acc_shadows)


if config.statistics.dataset == "CIFAR10":
    if config.statistics.type == "training_size":
        print("START STATS ON TRAINING SIZE WITH CIFAR10 : ", config.statistics.training_size_value)
        res_precision = np.zeros(len(config.statistics.training_size_value))
        res_recall = np.zeros(len(config.statistics.training_size_value))
        res_accuracy = np.zeros(len(config.statistics.training_size_value))
        # res_best_acc_target = np.zeros(len(config.statistics.training_size_value))
        # res_best_acc_shadows = np.zeros(len(config.statistics.training_size_value))
        res_precision_per_class = np.zeros((len(config.statistics.training_size_value), 10))
        res_recall_per_class = np.zeros((len(config.statistics.training_size_value), 10))
        res_accuracy_per_class = np.zeros((len(config.statistics.training_size_value), 10))
        for index_value_to_test, value_to_test in enumerate(config.statistics.training_size_value):
            config.set_subkey("general", "train_target_size", value_to_test)
            precision_general, recall_general, accuracy_general, precision_per_class, recall_per_class, accuracy_per_class = cifar10(config, path, index_value_to_test)
            res_precision[index_value_to_test] = precision_general
            res_recall[index_value_to_test] = recall_general
            res_accuracy[index_value_to_test] = accuracy_general
            #res_best_acc_target[index_value_to_test] = best_acc_target
            #res_best_acc_shadows[index_value_to_test] = best_acc_shadows
            res_precision_per_class[index_value_to_test] = precision_per_class
            res_recall_per_class[index_value_to_test] = recall_per_class
            res_accuracy_per_class[index_value_to_test] = accuracy_per_class
            plot_title = "CIFAR10"
            drawPlot(accuracy_per_class, precision_per_class, recall_per_class, plot_title, path)
            print("accuracy_per_class:", accuracy_per_class)
            print("precision_per_class:", precision_per_class)
            print("recall_per_class:", recall_per_class)
            np.savetxt(path + "/res_precision.csv", res_precision)
            np.savetxt(path + "/res_recall.csv", res_recall)
            np.savetxt(path + "/res_accuracy.csv", res_accuracy)
            #np.savetxt(path + "/res_best_acc_target.csv", res_best_acc_target)
            np.savetxt(path + "/res_recall_per_class.csv", res_recall_per_class)
            np.savetxt(path + "/res_precision_per_class.csv", res_precision_per_class)
            np.savetxt(path + "/res_accuracy_per_class.csv", res_accuracy_per_class)
            #np.savetxt(path + "/res_best_acc_shadows.csv", res_best_acc_shadows)
    if config.statistics.type == "number_shadow":
        print("START STATS ON NUMBER SHADOW  WITH CIFAR10 : ", config.statistics.number_shadow_value)
        res_precision = np.zeros(len(config.statistics.number_shadow_value))
        res_recall = np.zeros(len(config.statistics.number_shadow_value))
        res_accuracy = np.zeros(len(config.statistics.number_shadow_value))
        #res_best_acc_target = np.zeros(len(config.statistics.number_shadow_value))
        #res_best_acc_shadows = np.zeros(len(config.statistics.number_shadow_value))
        res_precision_per_class = np.zeros((len(config.statistics.number_shadow_value), 10))
        res_recall_per_class = np.zeros((len(config.statistics.number_shadow_value), 10))
        res_accuracy_per_class = np.zeros((len(config.statistics.number_shadow_value), 10))
        for index_value_to_test, value_to_test in enumerate(config.statistics.number_shadow_value):
            config.set_subkey("general", "number_shadow_model", value_to_test)
            precision_general, recall_general, accuracy_general, precision_per_class, recall_per_class, accuracy_per_class = cifar10(config, path, index_value_to_test)
            res_precision[index_value_to_test] = precision_general
            res_recall[index_value_to_test] = recall_general
            res_accuracy[index_value_to_test] = accuracy_general
            #res_best_acc_target[index_value_to_test] = best_acc_target
            #res_best_acc_shadows[index_value_to_test] = best_acc_shadows
            res_precision_per_class[index_value_to_test] = precision_per_class
            res_recall_per_class[index_value_to_test] = recall_per_class
            res_accuracy_per_class[index_value_to_test] = accuracy_per_class
            plot_title = "CIFAR10"
            drawPlot(accuracy_per_class, precision_per_class, recall_per_class, plot_title, path)
            np.savetxt(path + "/res_precision.csv", res_precision)
            np.savetxt(path + "/res_recall.csv", res_recall)
            np.savetxt(path + "/res_accuracy.csv", res_accuracy)
            #np.savetxt(path + "/res_best_acc_target.csv", res_best_acc_target)
            np.savetxt(path + "/res_recall_per_class.csv", res_recall_per_class)
            np.savetxt(path + "/res_precision_per_class.csv", res_precision_per_class)
            np.savetxt(path + "/res_accuracy_per_class.csv", res_accuracy_per_class)
            #np.savetxt(path + "/res_best_acc_shadows.csv", res_best_acc_shadows)
    if config.statistics.type == "overfitting":
        print("START STATS ON OVERFITTING WITH CIFAR10 : ", config.statistics.epoch_value)
        res_precision = np.zeros(len(config.statistics.epoch_value))
        res_recall = np.zeros(len(config.statistics.epoch_value))
        res_accuracy = np.zeros(len(config.statistics.epoch_value))
        #res_best_acc_target = np.zeros(len(config.statistics.epoch_value))
        #res_best_acc_shadows = np.zeros(len(config.statistics.epoch_value))
        res_precision_per_class = np.zeros((len(config.statistics.epoch_value), 10))
        res_recall_per_class = np.zeros((len(config.statistics.epoch_value), 10))
        res_accuracy_per_class = np.zeros((len(config.statistics.epoch_value), 10))
        for index_value_to_test, value_to_test in enumerate(config.statistics.epoch_value):
            config.set_subkey("learning", "epochs", value_to_test)
            precision_general, recall_general, accuracy_general, precision_per_class, recall_per_class, accuracy_per_class = cifar10(config, path, index_value_to_test)
            res_precision[index_value_to_test] = precision_general
            res_recall[index_value_to_test] = recall_general
            res_accuracy[index_value_to_test] = accuracy_general
            #res_best_acc_target[index_value_to_test] = best_acc_target
            #res_best_acc_shadows[index_value_to_test] = best_acc_shadows
            res_precision_per_class[index_value_to_test] = precision_per_class
            res_recall_per_class[index_value_to_test] = recall_per_class
            res_accuracy_per_class[index_value_to_test] = accuracy_per_class
            plot_title = "CIFAR10"
            drawPlot(accuracy_per_class, precision_per_class, recall_per_class, plot_title, path)
            np.savetxt(path + "/res_precision.csv", res_precision)
            np.savetxt(path + "/res_recall.csv", res_recall)
            np.savetxt(path + "/res_accuracy.csv", res_accuracy)
            #np.savetxt(path + "/res_best_acc_target.csv", res_best_acc_target)
            np.savetxt(path + "/res_recall_per_class.csv", res_recall_per_class)
            np.savetxt(path + "/res_precision_per_class.csv", res_precision_per_class)
            np.savetxt(path + "/res_accuracy_per_class.csv", res_accuracy_per_class)
            #np.savetxt(path + "/res_best_acc_shadows.csv", res_best_acc_shadows)

if config.statistics.dataset == "STL10":
    if config.statistics.type == "training_size":
        print("START STATS ON TRAINING SIZE WITH STL10 : ", config.statistics.training_size_value)
        res_precision = np.zeros(len(config.statistics.training_size_value))
        res_recall = np.zeros(len(config.statistics.training_size_value))
        res_accuracy = np.zeros(len(config.statistics.training_size_value))
        # res_best_acc_target = np.zeros(len(config.statistics.training_size_value))
        # res_best_acc_shadows = np.zeros(len(config.statistics.training_size_value))
        res_precision_per_class = np.zeros((len(config.statistics.training_size_value), 10))
        res_recall_per_class = np.zeros((len(config.statistics.training_size_value), 10))
        res_accuracy_per_class = np.zeros((len(config.statistics.training_size_value), 10))
        for index_value_to_test, value_to_test in enumerate(config.statistics.training_size_value):
            config.set_subkey("general", "train_target_size", value_to_test)
            precision_general, recall_general, accuracy_general, precision_per_class, recall_per_class, accuracy_per_class = stl10(config, path, index_value_to_test)
            res_precision[index_value_to_test] = precision_general
            res_recall[index_value_to_test] = recall_general
            res_accuracy[index_value_to_test] = accuracy_general
            #res_best_acc_target[index_value_to_test] = best_acc_target
            #res_best_acc_shadows[index_value_to_test] = best_acc_shadows
            res_precision_per_class[index_value_to_test] = precision_per_class
            res_recall_per_class[index_value_to_test] = recall_per_class
            res_accuracy_per_class[index_value_to_test] = accuracy_per_class
            plot_title = "STL10"
            drawPlot(accuracy_per_class, precision_per_class, recall_per_class, plot_title, path)
            np.savetxt(path + "/res_precision.csv", res_precision)
            np.savetxt(path + "/res_recall.csv", res_recall)
            np.savetxt(path + "/res_accuracy.csv", res_accuracy)
            #np.savetxt(path + "/res_best_acc_target.csv", res_best_acc_target)
            np.savetxt(path + "/res_recall_per_class.csv", res_recall_per_class)
            np.savetxt(path + "/res_precision_per_class.csv", res_precision_per_class)
            np.savetxt(path + "/res_accuracy_per_class.csv", res_accuracy_per_class)
            #np.savetxt(path + "/res_best_acc_shadows.csv", res_best_acc_shadows)
    if config.statistics.type == "number_shadow":
        print("START STATS ON NUMBER SHADOW  WITH STL10 : ", config.statistics.number_shadow_value)
        res_precision = np.zeros(len(config.statistics.number_shadow_value))
        res_recall = np.zeros(len(config.statistics.number_shadow_value))
        res_accuracy = np.zeros(len(config.statistics.number_shadow_value))
        #res_best_acc_target = np.zeros(len(config.statistics.number_shadow_value))
        #res_best_acc_shadows = np.zeros(len(config.statistics.number_shadow_value))
        res_precision_per_class = np.zeros((len(config.statistics.number_shadow_value), 10))
        res_recall_per_class = np.zeros((len(config.statistics.number_shadow_value), 10))
        res_accuracy_per_class = np.zeros((len(config.statistics.number_shadow_value), 10))
        for index_value_to_test, value_to_test in enumerate(config.statistics.number_shadow_value):
            config.set_subkey("general", "number_shadow_model", value_to_test)
            precision_general, recall_general, accuracy_general, precision_per_class, recall_per_class, accuracy_per_class = stl10(config, path, index_value_to_test)
            res_precision[index_value_to_test] = precision_general
            res_recall[index_value_to_test] = recall_general
            res_accuracy[index_value_to_test] = accuracy_general
            #res_best_acc_target[index_value_to_test] = best_acc_target
            #res_best_acc_shadows[index_value_to_test] = best_acc_shadows
            res_precision_per_class[index_value_to_test] = precision_per_class
            res_recall_per_class[index_value_to_test] = recall_per_class
            res_accuracy_per_class[index_value_to_test] = accuracy_per_class
            plot_title = "STL10"
            drawPlot(accuracy_per_class, precision_per_class, recall_per_class, plot_title, path)
            np.savetxt(path + "/res_precision.csv", res_precision)
            np.savetxt(path + "/res_recall.csv", res_recall)
            np.savetxt(path + "/res_accuracy.csv", res_accuracy)
            #np.savetxt(path + "/res_best_acc_target.csv", res_best_acc_target)
            np.savetxt(path + "/res_recall_per_class.csv", res_recall_per_class)
            np.savetxt(path + "/res_precision_per_class.csv", res_precision_per_class)
            np.savetxt(path + "/res_accuracy_per_class.csv", res_accuracy_per_class)
            #np.savetxt(path + "/res_best_acc_shadows.csv", res_best_acc_shadows)
    if config.statistics.type == "overfitting":
        print("START STATS ON OVERFITTING WITH STL10 : ", config.statistics.epoch_value)
        res_precision = np.zeros(len(config.statistics.epoch_value))
        res_recall = np.zeros(len(config.statistics.epoch_value))
        res_accuracy = np.zeros(len(config.statistics.epoch_value))
        #res_best_acc_target = np.zeros(len(config.statistics.epoch_value))
        #res_best_acc_shadows = np.zeros(len(config.statistics.epoch_value))
        res_precision_per_class = np.zeros((len(config.statistics.epoch_value), 10))
        res_recall_per_class = np.zeros((len(config.statistics.epoch_value), 10))
        res_accuracy_per_class = np.zeros((len(config.statistics.epoch_value), 10))
        for index_value_to_test, value_to_test in enumerate(config.statistics.epoch_value):
            config.set_subkey("learning", "epochs", value_to_test)
            precision_general, recall_general, accuracy_general, precision_per_class, recall_per_class, accuracy_per_class = stl10(config, path, index_value_to_test)
            res_precision[index_value_to_test] = precision_general
            res_recall[index_value_to_test] = recall_general
            res_accuracy[index_value_to_test] = accuracy_general
            #res_best_acc_target[index_value_to_test] = best_acc_target
            #res_best_acc_shadows[index_value_to_test] = best_acc_shadows
            res_precision_per_class[index_value_to_test] = precision_per_class
            res_recall_per_class[index_value_to_test] = recall_per_class
            res_accuracy_per_class[index_value_to_test] = accuracy_per_class
            plot_title = "STL10"
            drawPlot(accuracy_per_class, precision_per_class, recall_per_class, plot_title, path)
            np.savetxt(path + "/res_precision.csv", res_precision)
            np.savetxt(path + "/res_recall.csv", res_recall)
            np.savetxt(path + "/res_accuracy.csv", res_accuracy)
            #np.savetxt(path + "/res_best_acc_target.csv", res_best_acc_target)
            np.savetxt(path + "/res_recall_per_class.csv", res_recall_per_class)
            np.savetxt(path + "/res_precision_per_class.csv", res_precision_per_class)
            np.savetxt(path + "/res_accuracy_per_class.csv", res_accuracy_per_class)
            #np.savetxt(path + "/res_best_acc_shadows.csv", res_best_acc_shadows)


if config.statistics.dataset == "CIFAR100":
    if config.statistics.type == "training_size":
        print("START STATS ON TRAINING SIZE WITH CIFAR100 : ", config.statistics.training_size_value)
        res_precision = np.zeros(len(config.statistics.training_size_value))
        res_recall = np.zeros(len(config.statistics.training_size_value))
        res_accuracy = np.zeros(len(config.statistics.training_size_value))
        # res_best_acc_target = np.zeros(len(config.statistics.training_size_value))
        # res_best_acc_shadows = np.zeros(len(config.statistics.training_size_value))
        res_precision_per_class = np.zeros((len(config.statistics.training_size_value), 100))
        res_recall_per_class = np.zeros((len(config.statistics.training_size_value), 100))
        res_accuracy_per_class = np.zeros((len(config.statistics.training_size_value), 100))
        for index_value_to_test, value_to_test in enumerate(config.statistics.training_size_value):
            config.set_subkey("general", "train_target_size", value_to_test)
            precision_general, recall_general, accuracy_general, precision_per_class, recall_per_class, accuracy_per_class = cifar100(config, path, index_value_to_test)
            res_precision[index_value_to_test] = precision_general
            res_recall[index_value_to_test] = recall_general
            res_accuracy[index_value_to_test] = accuracy_general
            #res_best_acc_target[index_value_to_test] = best_acc_target
            #res_best_acc_shadows[index_value_to_test] = best_acc_shadows
            res_precision_per_class[index_value_to_test] = precision_per_class
            res_recall_per_class[index_value_to_test] = recall_per_class
            res_accuracy_per_class[index_value_to_test] = accuracy_per_class
            plot_title = "CIFAR100"
            drawPlot(accuracy_per_class, precision_per_class, recall_per_class, plot_title, path)
            np.savetxt(path + "/res_precision.csv", res_precision)
            np.savetxt(path + "/res_recall.csv", res_recall)
            np.savetxt(path + "/res_accuracy.csv", res_accuracy)
            #np.savetxt(path + "/res_best_acc_target.csv", res_best_acc_target)
            np.savetxt(path + "/res_recall_per_class.csv", res_recall_per_class)
            np.savetxt(path + "/res_precision_per_class.csv", res_precision_per_class)
            np.savetxt(path + "/res_accuracy_per_class.csv", res_accuracy_per_class)
            #np.savetxt(path + "/res_best_acc_shadows.csv", res_best_acc_shadows)
    if config.statistics.type == "number_shadow":
        print("START STATS ON NUMBER SHADOW  WITH CIFAR100 : ", config.statistics.number_shadow_value)
        res_precision = np.zeros(len(config.statistics.number_shadow_value))
        res_recall = np.zeros(len(config.statistics.number_shadow_value))
        res_accuracy = np.zeros(len(config.statistics.number_shadow_value))
        #res_best_acc_target = np.zeros(len(config.statistics.number_shadow_value))
        #res_best_acc_shadows = np.zeros(len(config.statistics.number_shadow_value))
        res_precision_per_class = np.zeros((len(config.statistics.number_shadow_value), 100))
        res_recall_per_class = np.zeros((len(config.statistics.number_shadow_value), 100))
        res_accuracy_per_class = np.zeros((len(config.statistics.number_shadow_value), 100))
        for index_value_to_test, value_to_test in enumerate(config.statistics.number_shadow_value):
            config.set_subkey("general", "number_shadow_model", value_to_test)
            precision_general, recall_general, accuracy_general, precision_per_class, recall_per_class, accuracy_per_class = cifar100(config, path, index_value_to_test)
            res_precision[index_value_to_test] = precision_general
            res_recall[index_value_to_test] = recall_general
            res_accuracy[index_value_to_test] = accuracy_general
            #res_best_acc_target[index_value_to_test] = best_acc_target
            #res_best_acc_shadows[index_value_to_test] = best_acc_shadows
            res_precision_per_class[index_value_to_test] = precision_per_class
            res_recall_per_class[index_value_to_test] = recall_per_class
            res_accuracy_per_class[index_value_to_test] = accuracy_per_class
            plot_title = "CIFAR100"
            drawPlot(accuracy_per_class, precision_per_class, recall_per_class, plot_title, path)
            np.savetxt(path + "/res_precision.csv", res_precision)
            np.savetxt(path + "/res_recall.csv", res_recall)
            np.savetxt(path + "/res_accuracy.csv", res_accuracy)
            #np.savetxt(path + "/res_best_acc_target.csv", res_best_acc_target)
            np.savetxt(path + "/res_recall_per_class.csv", res_recall_per_class)
            np.savetxt(path + "/res_precision_per_class.csv", res_precision_per_class)
            np.savetxt(path + "/res_accuracy_per_class.csv", res_accuracy_per_class)
            #np.savetxt(path + "/res_best_acc_shadows.csv", res_best_acc_shadows)
    if config.statistics.type == "overfitting":
        print("START STATS ON OVERFITTING WITH CIFAR100 : ", config.statistics.epoch_value)
        res_precision = np.zeros(len(config.statistics.epoch_value))
        res_recall = np.zeros(len(config.statistics.epoch_value))
        res_accuracy = np.zeros(len(config.statistics.epoch_value))
        #res_best_acc_target = np.zeros(len(config.statistics.epoch_value))
        #res_best_acc_shadows = np.zeros(len(config.statistics.epoch_value))
        res_precision_per_class = np.zeros((len(config.statistics.epoch_value), 100))
        res_recall_per_class = np.zeros((len(config.statistics.epoch_value), 100))
        res_accuracy_per_class = np.zeros((len(config.statistics.epoch_value), 100))
        for index_value_to_test, value_to_test in enumerate(config.statistics.epoch_value):
            config.set_subkey("learning", "epochs", value_to_test)
            precision_general, recall_general, accuracy_general, precision_per_class, recall_per_class, accuracy_per_class = cifar100(config, path, index_value_to_test)
            res_precision[index_value_to_test] = precision_general
            res_recall[index_value_to_test] = recall_general
            res_accuracy[index_value_to_test] = accuracy_general
            #res_best_acc_target[index_value_to_test] = best_acc_target
            #res_best_acc_shadows[index_value_to_test] = best_acc_shadows
            res_precision_per_class[index_value_to_test] = precision_per_class
            res_recall_per_class[index_value_to_test] = recall_per_class
            res_accuracy_per_class[index_value_to_test] = accuracy_per_class
            plot_title = "CIFAR100"
            drawPlot(accuracy_per_class, precision_per_class, recall_per_class, plot_title, path)
            np.savetxt(path + "/res_precision.csv", res_precision)
            np.savetxt(path + "/res_recall.csv", res_recall)
            np.savetxt(path + "/res_accuracy.csv", res_accuracy)
            #np.savetxt(path + "/res_best_acc_target.csv", res_best_acc_target)
            np.savetxt(path + "/res_recall_per_class.csv", res_recall_per_class)
            np.savetxt(path + "/res_precision_per_class.csv", res_precision_per_class)
            np.savetxt(path + "/res_accuracy_per_class.csv", res_accuracy_per_class)
            #np.savetxt(path + "/res_best_acc_shadows.csv", res_best_acc_shadows)
