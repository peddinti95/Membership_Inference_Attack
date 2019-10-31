import numpy as np
import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    #else:
        #print('Confusion matrix, without normalization')
    #print(cm)
    #plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    #plt.show()

def drawLossAcc(plot_result,plot_title,path):
    """accuracy and loss train/value graph"""
    plt.figure()
    plt.subplot(2,2,1)
    plt.cla()
    plt.bar(plot_result[0], color='#1a53ff')
    plt.ylabel('loss train')
    plt.subplot(2,2,2)
    plt.bar(plot_result[1], color='#1a53ff')
    plt.ylabel('accuracy train')
    plt.subplot(2,2,3)
    plt.bar(plot_result[2], color='#1a53ff')
    plt.ylabel('loss val')
    plt.xlabel('Epoch')
    plt.subplot(2, 2, 4)
    plt.bar(plot_result[3], color='#1a53ff')
    plt.ylabel('accuracy val')
    plt.xlabel('Epoch')
    plt.tight_layout()
    plot_path = path + "/fig_" + plot_title + "_target_model.png"
    plt.savefig(plot_path)

def drawPlot(accuracy_per_class,precision_per_class,recall_per_class,plot_title,path):
    """my code starts to draw plot graph"""
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.bar(accuracy_per_class, color='#1a53ff', width=0.8, color = ['red', 'green'])
    plt.title("Results on "+plot_title)
    plt.ylabel('Accuracy')
    plt.subplot(3, 1, 2)
    plt.bar(recall_per_class, color='#1a53ff', width=0.8, color = ['red', 'green'])
    plt.ylabel('Recall')
    plt.subplot(3, 1, 3)
    plt.bar(precision_per_class, color='#1a53ff', width=0.8,color = ['red', 'green'])
    plt.ylabel('Precision')
    plt.tight_layout()
    plot_path=path+"/fig_"+plot_title+".png"
    plt.savefig(plot_path)
