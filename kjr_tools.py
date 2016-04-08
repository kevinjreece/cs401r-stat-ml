import numpy as np
import random
import math

def splitData(all_data, percent_split):
	temp_data = np.copy(all_data)
	np.random.shuffle(temp_data)
	split_index = int(len(temp_data) * percent_split)

	first_split_data = temp_data[:split_index]

	second_split_data = temp_data[split_index:]
	return first_split_data, second_split_data

def calculateAttributeInformation(all_data, column):
	attribute_choices = list(set(all_data[:, column]))

	choice_sum = 0
	for choice in attribute_choices:
		choice_data = np.array(filter(lambda x: x[column] == choice, all_data))
		choice_count = len(choice_data)
		choice_ratio = choice_count / float(len(all_data))
		choice_sum += choice_ratio * calculateAttributeChoiceInformation(choice_data, column, choice)

	return choice_sum

def calculateAttributeChoiceInformation(choice_data, column, choice):
	labels = list(set(choice_data[:, -1]))
	
	label_sum = 0
	for label in labels:
		label_count = len(np.array(filter(lambda x: x[-1] == label, choice_data)))
		label_ratio = label_count / float(len(choice_data))
		label_partial_sum = label_ratio * math.log(label_ratio, 2)
		label_sum -= label_partial_sum

	return label_sum

def calculateAttributePurity(all_data, column):
	attribute_choices = list(set(all_data[:, column]))
	label_choices = list(set(all_data[:, -1]))

	majority_sum = 0
	for choice in attribute_choices:
		choice_data = np.array(filter(lambda x: x[column] == choice, all_data))
		majority_class = max(label_choices, key=list(choice_data[:, -1]).count)
		majority_count = len(list(filter(lambda x: x[-1] == majority_class, choice_data)))
		majority_sum += majority_count
	return float(majority_sum) / len(all_data)

def removeNPArrayFromList(l, arr):
    i = 0
    size = len(l)
    while i != size and not np.array_equal(l[i], arr):
        i += 1
    if i != size:
        l.pop(i)
    else:
        raise ValueError('nparray not found in list.')

def normalize(data, attributes):
    norm_data = np.copy(data)
    for i, (attr_name, attr_type) in enumerate(attributes[:-1]):
        if attr_type == u'REAL':
            col = norm_data[:, i]
            col = col.astype(np.float)
            col_min = np.nanmin(col)
            col_max = np.nanmax(col)
            col_ptp = col_max - col_min
            for j, x in enumerate(col):
                if x != np.nan and col_ptp != 0:
                    col[j] = (x - col_min) / col_ptp
            norm_data[:, i] = col
    return norm_data

def autolabel(rects, ax, precision):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                ('{0:.' + str(precision) + 'f}').format(float(height)),
                ha='center', va='baseline', fontsize=16)