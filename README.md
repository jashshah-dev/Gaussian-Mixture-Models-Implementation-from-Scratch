# Gaussian-Mixture-Models-Implementation-from-Scratch-and-model-comparisons


This repository contains a Python implementation of the Gaussian Mixture Model (GMM) and Poisson Mixture Model from scratch. The code allows for the comparison of GMM and K-means clustering algorithms, as well as the comparison between GMM, K-means, and K-means++ algorithms. The comparison is performed using the 130 Hospitals Diabetic Dataset from the US.

## Contents

- [Background](#background)
- [Features](#features)
- [Dataset](#dataset)
- [Implementation](#implementation)
- [Comparison Metric Used](#Comparison-Metric-Used)
- [Results](#results)


## Background

This project focuses on comparing Gaussian Mixture Model (GMM) and Poisson Mixture Model algorithms for clustering analysis. GMM is a probabilistic model that assumes data points are generated from a mixture of Gaussian distributions. In contrast, the Poisson Mixture Model is based on the assumption that data points follow a Poisson distribution.

## K-means Algorithm
The K-means algorithm is an iterative clustering algorithm that aims to partition a given dataset into K distinct clusters. The algorithm starts by randomly selecting K initial cluster centroids and assigns each data point to the nearest centroid. It then recalculates the centroid positions based on the assigned data points and repeats the process until convergence. The final result is a set of K clusters, each represented by its centroid.

While K-means is relatively simple and efficient, it has some limitations. One major drawback is its sensitivity to the initial centroid positions, which can lead to suboptimal clustering results. Additionally, it assumes that the clusters are spherical and have similar sizes, which may not always hold in real-world datasets.

## K-means++ Algorithm
K-means++ is an extension of the K-means algorithm that addresses the issue of selecting good initial centroids. The K-means++ algorithm introduces a more intelligent initialization step to improve the chances of finding a globally optimal solution.

Instead of randomly selecting the initial centroids, K-means++ follows a probabilistic approach. It starts by selecting the first centroid uniformly at random from the data points. Subsequent centroids are selected based on their distance from the already chosen centroids, with higher probabilities given to data points that are farther away from existing centroids. This initialization process reduces the likelihood of getting stuck in local optima and typically leads to better clustering results.

The K-means++ algorithm retains the iterative update step of K-means, where the centroids are recomputed based on the assigned data points. By combining smart initialization with the iterative optimization process, K-means++ tends to converge faster and provide more accurate cluster assignments compared to the standard K-means algorithm.

## GMM Model
The Gaussian Mixture Model (GMM) is a probabilistic model that represents a dataset as a mixture of several Gaussian distributions. GMM assumes that the data points are generated from a combination of underlying Gaussian distributions, with each component representing a distinct cluster.

The GMM algorithm involves two main steps: the Expectation step (E-step) and the Maximization step (M-step), which are iteratively performed until convergence.

In the E-step, GMM estimates the probabilities of each data point belonging to each Gaussian component, known as the responsibilities. These probabilities are computed using the current estimates of the Gaussian parameters: mean, covariance, and mixture weights.

In the M-step, GMM updates the estimates of the Gaussian parameters based on the computed responsibilities. It maximizes the likelihood of the observed data by adjusting the mean, covariance, and mixture weights to better fit the data points assigned to each Gaussian component.

One advantage of GMM is its flexibility in capturing complex patterns in the data, as each Gaussian component can model a different mode or cluster. GMM is capable of fitting data points that do not adhere to spherical or equal-sized cluster assumptions.

## Features

- Custom implementation of Gaussian Mixture Model from scratch
- Implementation of Poisson Mixture Model by modifying the distribution assumption to Poisson
- Implementation of Exponential Mixture Modelby modifying the distribution assumption to Exponential
- Comparison of GMM and K-means clustering algorithms
- Comparison of GMM, K-means, and K-means++ algorithms
- Evaluation of clustering results using the 130 Hospitals Diabetic Dataset from the US

## Dataset

The dataset used in this project is the 130 Hospitals Diabetic Dataset from the US. It contains information on diabetic patients, including patient demographics, medical history, and treatment details. The dataset provides a rich source of information for clustering analysis and algorithm comparison.

## Implementation

The Python code in this repository is organized as follows:

- `Gaussian_Mixture_Models.py`: Contains the implementation of the Gaussian Mixture Model from scratch.Also has a detailed overview with comments about comparison of different models

## Comparison Metrics Used

## Comparison Metrics
In order to assess the performance of different clustering algorithms, we utilize several comparison metrics that provide insights into the quality and characteristics of the obtained clusters. The following metrics are employed in our evaluation:

## Calinski-Harabasz Index
The Calinski-Harabasz index, also known as the Variance Ratio Criterion, measures the ratio of between-cluster dispersion to within-cluster dispersion. A higher index value indicates better-defined and well-separated clusters. It is calculated based on the dispersion of data points around their respective cluster centroids and the dispersion between different cluster centroids.

## Silhouette Score
The Silhouette score evaluates the quality of clustering by measuring the cohesion and separation of data points within and between clusters. It computes the average Silhouette coefficient for all data points, ranging from -1 to 1. A higher Silhouette score suggests that the clusters are well-separated and internally cohesive, while a lower score indicates overlapping or poorly separated clusters.

## Within Sum of Square Error (SSE)
The Within Sum of Square Error, also known as the inertia or distortion, quantifies the compactness of clusters. It measures the sum of squared distances between each data point and its assigned cluster centroid. A lower SSE value indicates that the data points within each cluster are closer to their respective centroids, indicating better cluster compactness.

## Davies-Bouldin Score
The Davies-Bouldin score assesses the clustering quality by considering both the within-cluster dispersion and the between-cluster separation. It measures the average similarity between each cluster and its most similar cluster, while also considering the cluster sizes. A lower Davies-Bouldin score indicates better-defined and well-separated clusters.

## Results

I have ran experiments for comparing Gaussian Mixture Models and Gaussian Mixture models with K
means ++ intialization.
 Other comparison is between K means ++ and Gaussian Mixture Models with K means plus plus
initialization


## GMM and GMM++
From the initial three graphs we can see that when we initialize Gaussian Mixture Models with
K means plus plus initalization the Within Sum of Square Errors reduces although the reduction
is not significant.
 However we can conclude that better initialization of means has a good effect on clustering
algorithm.
 
## K++ and GMM++
From graphs 4,5,6 we can see K means ++ performs better as it has lower within sum of square
errors and higher Calinski Harabaz score.

The probable reason why K means with better initialization works well could be:
When the clusters in the data are well-separated and clearly distinct, it is easier for K-means to
accurately assign each data point to its nearest centroid, resulting in a lower WSS. In contrast,

GMM may have more difficulty accurately assigning data points to clusters if the clusters are
overlapping or have complex structures, which can lead to a higher WSS.

Well defined cluster separation is visible from Calinski Harabaz score and davies bouldin score
Calinski-Harabasz Score (CHS) is a clustering evaluation metric used to measure the quality of
clustering solutions. It calculates the ratio of the between-cluster variance to the within-cluster
variance. The CHS is higher when the clusters are well separated and the within-cluster variance
is small, and lower when the clusters are overlapping and the within-cluster variance is large.

Davies-Bouldin Score (DBS) is a clustering evaluation metric used to measure the similarity
between clusters. It calculates the average similarity between each cluster and its most similar
cluster, based on the ratio of within-cluster and between-cluster distances. A lower DBS indicates
better clustering solutions with higher intra-cluster similarity and lower inter-cluster similarity.
Thus by the definition we can see that since davies bouldin score of K means is less than that of
GMM it is showing better results.

However choice for clustering depends on other factors as well which we need to consider
Comparison of All four methods


However Gaussian Mixture model when initialized with K means plus plus also gives better Davies
Bouldin Score as well as Calinski Harabaz Score as compared to K means ++
Thus we can conclude even though K means ++ reduces the within sum of square errors, GMM
with K means plus plus initialization have better cluster definition.
Thus the final selection of model also depends on other factors but K means ++ initialization
with PCA can serve as a better recipe.

![image](https://github.com/jashshah-dev/Gaussian-Mixture-Models-Implementation-from-Scratch/assets/132673402/6165a3b2-27c7-4ed8-86ba-84498ff783d2)
![image](https://github.com/jashshah-dev/Gaussian-Mixture-Models-Implementation-from-Scratch/assets/132673402/a66794eb-9315-4a9c-80e8-c0a89822ea4d)
![image](https://github.com/jashshah-dev/Gaussian-Mixture-Models-Implementation-from-Scratch/assets/132673402/c8a5350f-c2df-4eaa-929c-cff21d1a06c1)
![image](https://github.com/jashshah-dev/Gaussian-Mixture-Models-Implementation-from-Scratch/assets/132673402/c3e9a211-9382-4efc-a254-baa75c7fc44b)
![image](https://github.com/jashshah-dev/Gaussian-Mixture-Models-Implementation-from-Scratch/assets/132673402/81b39341-c349-426a-b069-d857fa9aea7f)









