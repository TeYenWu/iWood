import numpy as np
import matplotlib.pyplot as plt


def plot_and_save_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):

    if len(target_names) > 10:
        plt.rcParams.update({'font.size': 40})
    else:
        plt.rcParams.update({'font.size': 70})

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(20, 20), dpi = 120)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    plt.colorbar()

    # if target_names is not None:
    #     tick_marks = np.arange(len(target_names))
    #     plt.xticks(tick_marks, target_names, rotation=45)
    #     plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = cm * 100

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if normalize:
                if cm[i, j] == 100:
                    plt.text(j, i, "{:,}".format(cm[i, j]),
                             horizontalalignment="center",
                             color="white" if cm[i, j] > thresh else "black")
                else:
                    plt.text(j, i, "{:0.1f}".format(cm[i, j]),
                             horizontalalignment="center",
                             color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('./cm/confusion_matrix_'+str(title)+'.jpg')
    plt.clf()

cm = [[90, 10, 0, 0, 0, 0],
      [9, 88, 0, 0, 0, 3],
      [1, 0, 99, 0, 0, 0],
      [0, 0, 0, 96, 4, 0],
      [0, 0, 3, 3, 91, 3],
      [2, 0, 0, 0, 0, 98]]
cm = np.array(cm)
plot_and_save_confusion_matrix(cm, [], title = "cuttingboard_crossuser")



