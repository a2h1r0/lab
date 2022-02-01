import csv
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(__file__))


FILENAME = '../data/result_10_classes_dependent_20220131_140035.csv'
NUM_CLASSES = 10
OUTPUT = './confusion_matrix_' + str(NUM_CLASSES) + '_dependent'


answers_coffee, answers_dishwashing, answers_shampoo, answers_skinmilk, answers_tokkuri = [], [], [], [], []
predictions_coffee, predictions_dishwashing, predictions_shampoo, predictions_skinmilk, predictions_tokkuri = [], [], [], [], []


with open(FILENAME) as f:
    reader = csv.reader(f)

    for row in reader:
        if '(' not in row[0]:
            if 'coffee' in row[0]:
                answers_coffee.append(row[1])
                predictions_coffee.append(row[2])
            elif 'dishwashing' in row[0]:
                answers_dishwashing.append(row[1])
                predictions_dishwashing.append(row[2])
            elif 'shampoo' in row[0]:
                answers_shampoo.append(row[1])
                predictions_shampoo.append(row[2])
            elif 'skinmilk' in row[0]:
                answers_skinmilk.append(row[1])
                predictions_skinmilk.append(row[2])
            elif 'tokkuri' in row[0]:
                answers_tokkuri.append(row[1])
                predictions_tokkuri.append(row[2])


if NUM_CLASSES == 5:
    scale = ['Bottle A', 'Bottle B', 'Bottle C', 'Bottle D', 'Bottle E']
    labels = ['coffee', 'dishwashing', 'shampoo', 'skinmilk', 'tokkuri']

    answers = answers_coffee + answers_dishwashing + answers_shampoo + answers_skinmilk + answers_tokkuri
    predictions = predictions_coffee + predictions_dishwashing + predictions_shampoo + predictions_skinmilk + predictions_tokkuri

    sns.heatmap(pd.DataFrame(data=confusion_matrix(answers, predictions, labels=labels),
                             index=scale, columns=scale), annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Output', fontsize=18)
    plt.ylabel('Input', fontsize=18)
    plt.savefig(OUTPUT + '.eps', bbox_inches='tight', pad_inches=0)
    plt.savefig(OUTPUT + '.svg', bbox_inches='tight', pad_inches=0)
    plt.close()

else:
    if NUM_CLASSES == 2:
        scale = ['0%' + u'\u2013' + '90%', '90%' + u'\u2013' + '100%']
        labels = ['0-90', '90-100']
    elif NUM_CLASSES == 10:
        scale = ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%']
        labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100']

    sns.heatmap(pd.DataFrame(data=confusion_matrix(answers_coffee, predictions_coffee, labels=labels),
                             index=scale, columns=scale), annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Output', fontsize=18)
    plt.ylabel('Input', fontsize=18)
    plt.savefig(OUTPUT + '_coffee.eps', bbox_inches='tight', pad_inches=0)
    plt.savefig(OUTPUT + '_coffee.svg', bbox_inches='tight', pad_inches=0)
    plt.close()

    sns.heatmap(pd.DataFrame(data=confusion_matrix(answers_dishwashing, predictions_dishwashing, labels=labels),
                             index=scale, columns=scale), annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Output', fontsize=18)
    plt.ylabel('Input', fontsize=18)
    plt.savefig(OUTPUT + '_dishwashing.eps', bbox_inches='tight', pad_inches=0)
    plt.savefig(OUTPUT + '_dishwashing.svg', bbox_inches='tight', pad_inches=0)
    plt.close()

    sns.heatmap(pd.DataFrame(data=confusion_matrix(answers_shampoo, predictions_shampoo, labels=labels),
                             index=scale, columns=scale), annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Output', fontsize=18)
    plt.ylabel('Input', fontsize=18)
    plt.savefig(OUTPUT + '_shampoo.eps', bbox_inches='tight', pad_inches=0)
    plt.savefig(OUTPUT + '_shampoo.svg', bbox_inches='tight', pad_inches=0)
    plt.close()

    sns.heatmap(pd.DataFrame(data=confusion_matrix(answers_skinmilk, predictions_skinmilk, labels=labels),
                             index=scale, columns=scale), annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Output', fontsize=18)
    plt.ylabel('Input', fontsize=18)
    plt.savefig(OUTPUT + '_skinmilk.eps', bbox_inches='tight', pad_inches=0)
    plt.savefig(OUTPUT + '_skinmilk.svg', bbox_inches='tight', pad_inches=0)
    plt.close()

    sns.heatmap(pd.DataFrame(data=confusion_matrix(answers_tokkuri, predictions_tokkuri, labels=labels),
                             index=scale, columns=scale), annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Output', fontsize=18)
    plt.ylabel('Input', fontsize=18)
    plt.savefig(OUTPUT + '_tokkuri.eps', bbox_inches='tight', pad_inches=0)
    plt.savefig(OUTPUT + '_tokkuri.svg', bbox_inches='tight', pad_inches=0)
    plt.close()
