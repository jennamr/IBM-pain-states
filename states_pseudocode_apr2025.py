#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 13:04:05 2025

@author: jennareinen

This code serves as pseudocode and example code for the paper "Defining and validating a 
multidimensional digital metric of health states in chronic pain" by Reinen 
et al. Taking from specific lines in the Methods Section and in the Supplement, 
such code can be created and edited. 

This is intended to serve as a conceptual guide for our thought process and 
should be amended substantially when used with real data to accommodate the 
specifications of each study. 

"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import plotly.express as px

''' 
1. CREATE PSEUDO/EXAMPLE DATA & SEPARATE VALIDATION DATA 
Create pseudo-data comprised of x observations (here, x = 300) 
of y variables (here, y = 6) with different means and SD. 
They should have participant numbers and dates assigned to each.
It is possible some participants respond more than once. 
''' 

# Define means and standard deviations for each variable
means = {
    'Participant': 50,
    'Date': 20220101,
    'Variable1': 50,
    'Variable2': 75,
    'Variable3': 25,
    'Variable4': 100,
    'Variable5': 30,
    'Variable6': 60
}

# Define standard deviations for each variable
stds = {
    'Participant': 10,
    'Date': 1,
    'Variable1': 5,
    'Variable2': 10,
    'Variable3': 3,
    'Variable4': 15,
    'Variable5': 5,
    'Variable6': 10
}

# Create the dataset
data = {
    'Participant': np.random.normal(means['Participant'], stds['Participant'], 300).astype(int),
    'Date': np.random.normal(means['Date'], stds['Date'], 300).astype(int),
    'Variable1': np.random.normal(means['Variable1'], stds['Variable1'], 300),
    'Variable2': np.random.normal(means['Variable2'], stds['Variable2'], 300),
    'Variable3': np.random.normal(means['Variable3'], stds['Variable3'], 300),
    'Variable4': np.random.normal(means['Variable4'], stds['Variable4'], 300),
    'Variable5': np.random.normal(means['Variable5'], stds['Variable5'], 300),
    'Variable6': np.random.normal(means['Variable6'], stds['Variable6'], 300)
}

# Convert the data to a pandas DataFrame 
data_df = pd.DataFrame(data)

# Print the first few rows of the DataFrame
print(data_df.head())

# Hold out one variable for validation 
validation_df = data_df[['Participant', 'Date', 'Variable6']]
data_df = data_df.drop('Variable6', axis=1)


''' 
2. CLEAN DATA 
- Here is where the data could be cleaned due for study-specific reasons (not shown)
- Data going into a k-means model can be normalized and scaled   
'''

# Scale features 
features = data_df.drop(['Participant', 'Date'], axis=1)
scaler = MinMaxScaler(feature_range=(0, 1))
features_scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

''' 
4. DETERMINE OPTIMAL K USING ELBOW METHOD 
- Other approaches, such as calculating Silhouette Scores or using 
Consensus Clustering, could be used but are not shown here. 
''' 

# List to store the SSD for each k
ssd = []

# Range of k values to test
k_values = range(1, 11)

# Loop over each k value
for k in k_values:
    # Fit the KMeans model
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(features_scaled)
    
    # Append the sum of squared distances (SSD)
    ssd.append(kmeans.inertia_)

# Plot the Elbow
plt.plot(k_values, ssd, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum of squared distances')
plt.title('Elbow Method showing the optimal k')
plt.show()

# Let's say here, hypothetically k = 4

''' 
5. GENERATE MODEL FOR OPTIMAL K AND VISUALIZE THE CLUSTERS BY FEATURE
''' 

# Create a k-means model for k = 4
kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(features_scaled)
# Assuming kmeans is your fitted KMeans model
cluster_centers = kmeans.cluster_centers_
cluster_assignments = kmeans.labels_

# Number of variables
n = len(cluster_centers[0])

# Create a bar chart for each cluster
fig, axs = plt.subplots(4, 1, figsize=(8, 12))
X = features_scaled.to_numpy()

for i in range(4):
    cluster_indices = np.where(cluster_assignments == i)[0]
    cluster_data = X[cluster_indices]
    cluster_means = np.mean(cluster_data, axis=0)
    axs[i].bar(range(n), cluster_means, color=['r', 'g', 'b', 'y'])
    axs[i].set_xlabel('Feature')
    axs[i].set_ylabel('Mean Value')
    axs[i].set_title(f'Cluster {i + 1}')
    axs[i].set_xticks(range(n))
    axs[i].set_xticklabels([f'Feature {j + 1}' for j in range(n)])

plt.tight_layout()
plt.show()

# If the visualization make sense, labels can be applied 

'''
6.  CALCULATE CLUSTER RELATIONSHIP TO VALIDATION DATA 
'''

# Create a validation data frame: this includes, for every observation, \
# the participant ID, the date, the values, as well as a cluster metric \
# (possibly centroid distance) from each cluster in the optimal cluster model. 

kmeans_df = features_scaled
centroids = kmeans.cluster_centers_
num_clusters = centroids.shape[0]
distances = []

for i in range(num_clusters):
    centroid = centroids[i]
    distances_to_centroid = np.linalg.norm(kmeans_df - centroid, axis=1)
    distances.append(distances_to_centroid)
    
# Calculate correlation between held out validation data and distances 
distances_df = pd.DataFrame(distances).T
corrs1 = distances_df[[0,1,2,3]].corrwith(validation_df['Variable6'])
print(corrs1)

# This can be used to rank the clusters, or determine model is not usable.      
