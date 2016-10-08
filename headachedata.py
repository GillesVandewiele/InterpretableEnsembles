import random
from pandas import DataFrame
import numpy as np
import pandas as pd
from pymongo import MongoClient
from datetime import date, datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib
from scipy.cluster.hierarchy import fclusterdata
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

from constructors.c45orangeconstructor import C45Constructor
from constructors.cartconstructor import CARTConstructor
from constructors.questconstructor import QuestConstructor


class DataCollector(object):
    def __init__(self):
        pass

    @staticmethod
    def load_data_from_db(host, port, dbname):
        client = MongoClient(host, port)
        db = client[dbname]
        return db

    @staticmethod
    def calculate_age(born):
        today = date.today()
        return today.year - born.year - ((today.month, today.day) < (born.month, born.day))

    @staticmethod
    def plot_pie_chart(values, labels, title):
        # The slices will be ordered and plotted counter-clockwise.
        my_norm = matplotlib.colors.Normalize(0, 1)  # maps your data to the range [0, 1]
        my_cmap = matplotlib.cm.get_cmap('coolwarm')
        # print my_norm(values)
        fig = plt.figure()
        fig.suptitle(title, fontsize=25)

        matplotlib.rcParams['font.size'] = 18
        plt.pie(values, labels=labels, colors=my_cmap(my_norm(values)),
                autopct='%1.1f%%')
        # Set aspect ratio to be equal so that pie is drawn as a circle.
        plt.axis('equal')
        F = plt.gcf()
        Size = F.get_size_inches()
        F.set_size_inches(Size[0] * 1.25, Size[1] * 1.75, forward=True)
        plt.show()


database = DataCollector.load_data_from_db('localhost', 9000, 'CHRONIC')

# Get all collections from the database
patients = database['patient']
drugs = database['drug']
headaches = database['headache']
medicines = database['medicine']
symptoms = database['symptom']
triggers = database['trigger']

####################################################
#       Read the patient data into dataframe       #
####################################################
patient_column_names = ['id', 'age', 'sex', 'relation', 'employment', 'diagnosis']
patients_list = []
for patient in patients.find({}):
    # print patient
    patient_list = [patient['patientID'], patient['birthDate'], patient['isMale'], patient['relation'],
                    patient['isEmployed']]
    if 'diagnoseID' in patient:
        patient_list.append(patient['diagnoseID'])
    else:
        patient_list.append(-1)
    patients_list.append(patient_list)

####################################################
#   Map strings to integers, fill missing values   #
####################################################
patient_df = DataFrame(patients_list, columns=patient_column_names)
patient_df['age'] = [DataCollector.calculate_age(datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ")) if x != "null"
                     else np.NaN for x in patient_df['age']]
patient_df.age.replace(np.NaN, patient_df["age"].mean(), inplace=True)
patient_df['age'] = patient_df['age'].astype(int)
patient_df = patient_df[patient_df.id > 10]  # All patients with id higher than 10 are test accounts
patient_df = patient_df[patient_df.diagnosis != -1]
patient_df = patient_df.reset_index(drop=True)
diagnose_mapping = {1: "MIGRAINE W/ AURA", 2: "MIGRAINE W/O AURA", 3: "CLUSTER", 4: "TENSION"}
diagnose_mapping_reverse = {"MIGRAINE W/ AURA": 1, "MIGRAINE W/O AURA": 2, "CLUSTER": 3, "TENSION": 4}
patient_df['sex'] = patient_df['sex'].map(lambda x: "MALE" if x else "FEMALE")
patient_df['employment'] = patient_df['employment'].map(lambda x: "EMPLOYED" if x else "UNEMPLOYED")
patient_df['diagnosis'] = patient_df["diagnosis"].map(diagnose_mapping)

###################################################
#           Plot some demographic plots           #
###################################################

def get_distribution(values):
    distribution = {}
    for value in values:
        if value not in distribution:
            distribution[value] = 1
        else:
            distribution[value] += 1

    total_sum = np.sum(distribution.values())
    for value in distribution:
        distribution[value] = float(distribution[value]) / float(total_sum)

    return distribution

categorical_columns = ['sex', 'relation', 'employment', 'diagnosis']
for column in categorical_columns:
    col_distribution = get_distribution(patient_df[column].values)
    DataCollector.plot_pie_chart(col_distribution.values(), col_distribution.keys(),
                                 'Distribution of the ' + column + ' in the headache dataset')

n, bins, patches = plt.hist(patient_df['age'], 5, normed=0, facecolor='blue', alpha=0.75)
plt.xlabel('Age')
plt.ylabel('Absolute amount')
plt.title('Distribution of the age in the headache dataset')
plt.show()

##########################################################
#               Do some mapping                          #
##########################################################

relation_mapping = {"VRIJGEZEL": 0, "IN RELATIE": 1, "GETROUWD": 2}
patient_df['relation'] = patient_df['relation'].map(relation_mapping)
patient_df['relation'] = patient_df.relation.apply(lambda x: x if not pd.isnull(x) else 0)
patient_df['sex'] = patient_df['sex'].map(lambda x: 1 if "MALE" else 0)
patient_df['employment'] = patient_df['employment'].map(lambda x: 1 if "EMPLOYED" else 0)
patient_df['diagnosis'] = patient_df["diagnosis"].map(diagnose_mapping_reverse)
# print patient_df
# print patient_df.describe()

####################################################
#      Read the headache data into dataframe       #
#   We process the strings already in datatypes    #
####################################################
headache_locations = ["frontal_right", "frontal_mid", "frontal_left", "parietal_right", "parietal_mid",
                      "parietal_left", "temporal_right", "temporal_left", "occipital_right", "occipital_mid",
                      "occipital_left", "cervical_right", "cervical_mid", "cervical_left", "orbital_right",
                      "orbital_left", "mandibular_left", "mandibular_right", "maxillar_right", "maxillar_left"]
headache_column_names = ['id', 'intensities', 'end', 'patientID', 'symptomIDs', 'triggerIDs', 'locations']
headaches_list = []
for i in range(len(patient_df)):
    for headache in headaches.find({"patientID": patient_df.iloc[i, :]['id']}):
        row = [headache['headacheID']]
        intensity_dict = {}
        for intensity in headache['intensityValues']:
            intensity_dict[datetime.strptime(intensity['key'], "%Y-%m-%dT%H:%M:%S.%fZ")] = int(intensity['value'])
        row.append(intensity_dict)
        # Missing value for end: add 2 hours
        end_time = ""
        if headache['end'] == "null" or datetime.strptime(headache['end'], "%Y-%m-%dT%H:%M:%S.%fZ") < \
                sorted(list(intensity_dict.keys()))[-1]:
            # interpolleer
            # print "interpolleer"
            if len(intensity_dict.keys()) < 2:
                end_time = sorted(list(intensity_dict.keys()))[0] + timedelta(hours=2)
            else:
                # end_time = sorted(list(intensity_dict.keys()))[0] + timedelta(hours=2)
                diff_time = abs(sorted(list(intensity_dict.keys()))[-1]-sorted(list(intensity_dict.keys()))[-2])
                diff_value = abs(intensity_dict[sorted(list(intensity_dict.keys()))[-1]] - intensity_dict[sorted(list(intensity_dict.keys()))[-2]])
                last_value = intensity_dict[sorted(list(intensity_dict.keys()))[-2]]
                rico = diff_time/(diff_value+1)
                add = last_value*rico

                end_time = sorted(list(intensity_dict.keys()))[-1]+add

                row.extend([end_time,headache['patientID'], headache['symptomIDs'], headache['triggerIDs']])
        else:
            end_time = datetime.strptime(headache['end'], "%Y-%m-%dT%H:%M:%S.%fZ")

            row.extend([end_time,
                        headache['patientID'], headache['symptomIDs'], headache['triggerIDs']])
        sorted(list(intensity_dict.keys()))
        # print end_time
        location_dict = {}
        for location in headache['locations']:
            location_dict[location['key']] = location['value']
        row.append(location_dict)
        headaches_list.append(row)

headache_df = DataFrame(headaches_list, columns=headache_column_names)
# print headache_df
# print headache_df.describe()

####################################################
#       Read the symptom data into dataframe       #
####################################################
symptom_column_names = ['id', 'name']
symptoms_list = []
for symptom in symptoms.find({}):
    print symptom
    symptom_list = [symptom['symptomID'], symptom['name']]
    # if 'diagnoseID' in patient:
    #     patient_list.append(patient['diagnoseID'])
    # else:
    #     patient_list.append(-1)
    symptoms_list.append(symptom_list)

####################################################
#       Read the trigger data into dataframe       #
####################################################
trigger_column_names = ['id', 'name']
triggers_list = []
for trigger in triggers.find({}):
    print trigger
    trigger_list = [trigger['triggerID'], trigger['name']]
    # if 'diagnoseID' in patient:
    #     patient_list.append(patient['diagnoseID'])
    # else:
    #     patient_list.append(-1)
    triggers_list.append(trigger_list)

####################################################
#    Now that we have all required information,    #
#       we can make a features dataframe           #
####################################################

data_list = []

for i in range(len(patient_df)):
    vector = []
    # All patient demographic attributes are features
    vector.extend(patient_df.iloc[i, :].values)

    # Count number of headaches for a patient
    vector.append(len(headache_df[headache_df.patientID == patient_df.iloc[i, :]['id']]))
    filtered_df = headache_df[headache_df.patientID == patient_df.iloc[i, :]['id']]
    intensity_values = []
    durations = []
    location_freq_dict = {}
    for location in headache_locations:
        location_freq_dict[location] = 0

    symptom_freq_dict = {}

    for symptom in symptoms_list:
        symptom_freq_dict[int(symptom[0])] = 0.0

    trigger_freq_dict = {}

    for trigger in triggers_list:
        trigger_freq_dict[int(trigger[0])] = 0.0

    for _headache in range(len(filtered_df)):
        headache = filtered_df.iloc[_headache, :]
        # print headache['intensities']
        intensity_values.extend(headache['intensities'].values())

        duration = (headache['end'] - sorted(list(headache['intensities'].keys()))[0]).total_seconds()
        if duration < 0:
            duration = 7200
        timestamps_keys = sorted(list(headache['intensities'].items()))
        # print timestamps_keys

        # TODO: interpolate  sorted(list(headache['intensities'].items()))
        durations.append(duration)
        for location in headache['locations'].items():
            location_freq_dict[location[0]] += location[1]

        for symptomID in headache['symptomIDs']:
            symptom_freq_dict[symptomID] += 1

        for triggerID in headache['triggerIDs']:
            trigger_freq_dict[triggerID] += 1

        print headache

    totalsymptoms = sum(symptom_freq_dict.values())
    print totalsymptoms
    for symptomID in symptom_freq_dict.keys():
        if totalsymptoms != 0:
            symptom_freq_dict[symptomID] = float(symptom_freq_dict[symptomID]) / float(totalsymptoms)
        else:
            symptom_freq_dict[symptomID] = 0

    totaltriggers = sum(trigger_freq_dict.values())
    print totaltriggers
    for triggerID in trigger_freq_dict.keys():
        if totaltriggers != 0:
            trigger_freq_dict[triggerID] = float(trigger_freq_dict[triggerID]) / float(totaltriggers)
        else:
            trigger_freq_dict[triggerID] = 0

    # Intensity value mean and max
    vector.append(np.mean(intensity_values))
    vector.append(np.max(intensity_values) if len(intensity_values) else np.NaN)

    # Duration mean, max and min
    vector.append(np.mean(durations))
    vector.append(np.max(durations) if len(durations) else np.NaN)
    vector.append(np.min(durations) if len(durations) else np.NaN)

    # Relative frequency of all intensity values (0, 1, .., 10)
    vector.extend(np.histogram(intensity_values, bins=range(12), normed=True)[0])

    # Relative frequency for each location
    total_sum = sum(location_freq_dict.values())
    for location in headache_locations:
        if total_sum > 0:
            vector.append(location_freq_dict[location] / total_sum)
        else:
            vector.append(0)

    # Relative frequency for each symptom
    for symptom in symptoms_list:
        vector.append(symptom_freq_dict[symptom[0]])

    # Relative frequency for each trigger
    for trigger in triggers_list:
        vector.append(trigger_freq_dict[trigger[0]])

    data_list.append(vector)

intensity_names = []
for i in range(11):
    intensity_names.append("intensity_" + str(i))
columns = ["id", "age", "sex", "relation", "employment", "diagnosis", "headacheCount", "meanIntensity", "maxIntensity",
           "meanDuration", "maxDuration", "minDuration"]
columns.extend(intensity_names)
columns.extend(headache_locations)
columns.extend([x[1] + "_freq" for x in symptoms_list])
columns.extend([x[1] + "_freq" for x in triggers_list])

data_df = DataFrame(data_list, columns=columns)
data_df = data_df[data_df.headacheCount > 1]
data_df.reset_index(drop=True)

print data_df
features_df = data_df.copy()
features_df = features_df.dropna()
pca = PCA(n_components=3)
pca.fit(features_df)

transformed_features = []
for i in range(len(features_df)):
    feature_vector = features_df.iloc[i, :]
    transformed_feature = [feature_vector['id'], feature_vector['diagnosis']]
    transformed_feature.extend(*pca.transform(feature_vector))
    transformed_features.append(transformed_feature)

# print transformed_features
transformed_features_df = DataFrame(transformed_features, columns=["id", "diagnosis", "pca_1", "pca_2", "pca_3"])
# print transformed_features_df

fig = plt.figure()

ax = Axes3D(fig)

# ax.scatter(transformed_features_df['pca_1'], transformed_features_df['pca_2'], transformed_features_df['pca_3'])
colors = ["red", "blue", "green", "yellow"]

for i in range(len(transformed_features_df)):
    feature_vector = transformed_features_df.iloc[i, :]
    ax.scatter(feature_vector['pca_1'], feature_vector['pca_2'], feature_vector['pca_3'],
               c=colors[int(feature_vector["diagnosis"]) - 1], s=50)
    ax.text(feature_vector['pca_1'], feature_vector['pca_2'], feature_vector['pca_3'], feature_vector['id'])

plt.show()

# data_df = data_df.dropna()
#
# clusters = fclusterdata(data_df[["meanIntensity", "meanDuration"]], 0.1, criterion="distance")
# print clusters
# label_df = DataFrame()
# label_df["cat"] = features_df["diagnosis"]
#
# features_df = features_df.drop("diagnosis", axis=1)
# features_df = features_df.drop("id", axis=1)
#
# best_features_boruta = boruta_py_feature_selection(features_df.values, label_df['cat'].tolist(), columns,
#                                                    verbose=True, percentile=80, alpha=0.1)
#
#
# num_features_boruta = len(best_features_boruta)
#
# new_features_rf = DataFrame()
# new_features_boruta = DataFrame()
#
#
# for k in range(num_features_boruta):
#     new_features_boruta[columns[best_features_boruta[k]]] = features_df[columns[best_features_boruta[k]]]
#
# features_df_boruta = new_features_boruta
#
#
# cart = CARTConstructor(min_samples_split=1)
# tree = cart.construct_tree(new_features_boruta, labels=label_df)
# tree.visualise("./test.pdf")