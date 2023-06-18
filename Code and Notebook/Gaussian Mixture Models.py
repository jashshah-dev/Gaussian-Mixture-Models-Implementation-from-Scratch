#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
df=pd.read_csv('diabetic_data.csv')


# In[4]:


import numpy as np
df.replace({'?':np.nan},inplace=True)
df1=pd.DataFrame(df.isna().sum())
df1=df1.reset_index()
df1.columns=['Column_Names','Count_of_Nan_Values']
df2=df1[df1['Count_of_Nan_Values']!=0].sort_values(by=['Count_of_Nan_Values'],ascending=False)
df2['Percentage_of_NAN']=df2['Count_of_Nan_Values']/len(df)*100
print('The Nan Value columns with percentage are as follows')
print(df2)


# In[5]:


#Dropping columns with more than 40 percent null values
df.drop(['weight','payer_code','medical_specialty'],axis=1,inplace=True)
#Changing the readmitted column
df['readmitted'] = df['readmitted'].replace({'>30':1,'<30':1,'NO':0})
#Replacing Age with mean
df['age'] = df['age'].replace({'[70-80)': 75, '[60-70)': 65, '[50-60)': 55, '[80-90)': 85, '[40-50)': 45, '[30-40)': 35, '[90-100)': 95, '[20-30)': 25, '[10-20)': 15, '[0-10)': 5})


# In[6]:


df_diabetes=df.copy()
df_diabetes.drop(columns=['encounter_id','patient_nbr'],axis=1,inplace=True)


# In[7]:


imbalanced_data=['examide','metformin-rosiglitazone','metformin-pioglitazone','glimepiride-pioglitazone','glipizide-metformin','glyburide-metformin','citoglipton','tolazamide','troglitazone','miglitol','acarbose','tolbutamide','acetohexamide','chlorpropamide','nateglinide','repaglinide']
df_diabetes.drop(columns=imbalanced_data,inplace=True)


# In[11]:


import swifter


# In[12]:


#Label Encoding in data where there is a ordinality
from sklearn.preprocessing import LabelEncoder
ordinal_columns=['max_glu_serum', 'A1Cresult',
       'metformin', 'glimepiride', 'glipizide', 'glyburide', 'pioglitazone',
       'rosiglitazone', 'insulin', 'change', 'diabetesMed']
df_diabetes[['max_glu_serum', 'A1Cresult',
       'metformin', 'glimepiride', 'glipizide', 'glyburide', 'pioglitazone',
       'rosiglitazone', 'insulin', 'change', 'diabetesMed']] = df_diabetes[['max_glu_serum', 'A1Cresult',
       'metformin', 'glimepiride', 'glipizide', 'glyburide', 'pioglitazone',
       'rosiglitazone', 'insulin', 'change', 'diabetesMed']].swifter.apply(LabelEncoder().fit_transform)


# In[13]:


df_diabetes = df_diabetes.drop(df_diabetes.loc[df_diabetes["gender"]=="Unknown/Invalid"].index, axis=0)


# In[14]:


ordinal_columns=['gender','race']
one_hot = pd.get_dummies(df_diabetes[['gender','race']])
df_diabetes=pd.concat([df_diabetes,one_hot],axis=1)


# In[15]:


df_diabetes.drop(columns=['diag_1','diag_2','diag_3','gender','race'],inplace=True)


# In[16]:


df_diabetes_final=df_diabetes.copy()


# In[17]:


df_diabetes_final.shape


# # Scaling the Dataframe

# In[44]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df=pd.DataFrame(df_diabetes_final)
scaler.fit(df)
diabetes_scaled=scaler.transform(df)
df_diabetes_scaled=pd.DataFrame(diabetes_scaled)
df_sample_scaled=df_diabetes_scaled.copy()
df_sample_scaled=df_sample_scaled.head(10)
#df_diabetes_scaled=df_diabetes_scaled.drop(index=index)
df_diabetes_scaled.shape


# # Starting the code for GMM

# In[126]:


def complex_euclidean_distance(u, v):
    return np.sqrt(np.sum(np.abs(u - v) ** 2))
def wcss_emm(input_dataframe,labels_array,no_of_clusters):
    from scipy.spatial.distance import cdist
    input_dataframe_clustered=input_dataframe.copy()
    input_dataframe_clustered['Labels']=labels_array
    new_centroids=input_dataframe_clustered.groupby('Labels').mean()
    new_centroids=new_centroids.T
    total_error=[]
    no_of_clusters_array=np.array(labels_array)
    no_of_clusters=np.unique(no_of_clusters_array)
    for cluster in no_of_clusters:
        df_data_label_cluster=input_dataframe_clustered[input_dataframe_clustered['Labels']==cluster]
        df_data_label_cluster=df_data_label_cluster.drop('Labels',axis=1)
        centroids=pd.DataFrame(new_centroids[cluster])
        euclidean_distance=cdist(df_data_label_cluster,centroids.T,metric=complex_euclidean_distance)
        total_error.append(sum(euclidean_distance))
    return round(float(''.join(map(str, sum(total_error)))),3)    

def silheoutte_score(input_dataframe,labels):
    from sklearn.metrics import silhouette_score
    sample_size=5000
    sample_indices=np.random.choice(input_dataframe.shape[0], size=sample_size, replace=False)
    sample_data = input_dataframe[subset_indices]
    silhouette_avg= silhouette_score(sample_data,labels[sample_indices], metric='cosine')
    return silhouette_avg

def Calinski_Harbaz_score(input_dataframe,labels):
    from sklearn.metrics import calinski_harabasz_score
    chs=calinski_harabasz_score(input_dataframe,labels)
    return chs

def davies_bouldin_score(input_dataframe,labels):
    from sklearn.metrics import davies_bouldin_score
    dbs=davies_bouldin_score(input_dataframe,labels)
    return dbs


# In[127]:


from scipy.stats import multivariate_normal
import numpy as np

def initialization_of_GMM(input_dataframe,no_of_clusters):
    '''
    The function takes scaled dataframe as input and initializes the GMM means,Covariances,and Weights
    '''
    input_dataframe_values = input_dataframe.values
    row, column = input_dataframe_values.shape
    # Randomly initialize means vector
    means_vector = input_dataframe_values[np.random.choice(input_dataframe_values.shape[0], no_of_clusters, replace=False), :]
    # Initialize covariance matrices for each cluster
    covariances_vector = np.array([np.eye(column)] * no_of_clusters)
    # Initialize weights from uniform distribution
    weights_vector = np.ones(no_of_clusters) / no_of_clusters
    return means_vector,covariances_vector,weights_vector
    
    
def fit_Guassian_mixture_models(input_dataframe,no_of_clusters,max_no_of_iterations,threshold):
    input_dataframe_values = input_dataframe.values
    row, column = input_dataframe_values.shape
    means,covariances,weights=initialization_of_GMM(input_dataframe,no_of_clusters)
    iteration = 0
    previous_log_likelihood_scalar=0
    while iteration < max_no_of_iterations:
       
        new_log_likelihood = 0
        for index in range(no_of_clusters):
            try:
                epsilon_weight=1e-6
                cov_inv = np.linalg.pinv(covariances[index] + np.diag(np.ones(covariances[index].shape[0]) * epsilon_weight))
                new_log_likelihood=new_log_likelihood+weights[index]*multivariate_normal.logpdf(input_dataframe_values,means[index], cov_inv)
            except np.linalg.LinAlgError as e:
                continue
        new_log_likelihood_scalar=np.sum(new_log_likelihood)
        
        '''
        Calculating percentage change
        '''
        if np.abs(((np.abs(new_log_likelihood_scalar-previous_log_likelihood_scalar)/new_log_likelihood_scalar)*100))<threshold:
            print("The input Threshold was {}".format(threshold))
            print("The calculated threshold is {}".format(np.abs(((np.abs(new_log_likelihood_scalar-previous_log_likelihood_scalar)/new_log_likelihood_scalar)*100))))
            break
        #else:
            #print("The input Threshold was {}".format(threshold))
            #print("The calculated threshold is {}".format(np.abs(((np.abs(new_log_likelihood_scalar-previous_log_likelihood_scalar)/new_log_likelihood_scalar)*100))))
        previous_log_likelihood_scalar=new_log_likelihood_scalar
        posterior_probabilities = np.zeros((len(input_dataframe_values),no_of_clusters))
        for index in range(no_of_clusters):
            try:
                cov_inv = np.linalg.pinv(covariances[index],rcond=1e-10)
            except np.linalg.LinAlgError as e:
                continue
            try:
                posterior_probabilities[:,index] = weights[index] * multivariate_normal.pdf(input_dataframe_values, means[index], cov_inv)
            except np.linalg.LinAlgError as e:
                continue
        posterior_probabilities/=np.sum(posterior_probabilities, axis=1, keepdims=True)

        
        
        for j in range(no_of_clusters):
            weighted_sum = np.zeros((1, means.shape[1]))
            sum_posterior = 0.0
            for i in range(row):
                weighted_sum += posterior_probabilities[i][j] * input_dataframe_values[i]
                sum_posterior += posterior_probabilities[i][j]
            means[j] = weighted_sum/sum_posterior
            difference = input_dataframe_values - means[j]
            covariances[j] = np.dot((difference * posterior_probabilities[:, j][:, np.newaxis]).T, difference) / np.sum(posterior_probabilities[:, j])
            covariances[j] += np.diag(np.ones(column) * 1e-6)
            weights[j] = np.mean(posterior_probabilities[:, j])
        
        
        
        iteration += 1
        
    return means,posterior_probabilities


# # Running the code Multiple Times

# In[749]:


expectation_maximization_statistics=[]
for no_of_clusters in range(2,6):
    print(no_of_clusters)
    for no_of_experiments in range(1,21):
        print(no_of_experiments)
        means,posterior_probabilities=fit_Guassian_mixture_models(df_diabetes_scaled,no_of_clusters,100,1)
        cluster_labels_original=np.array(pd.DataFrame(posterior_probabilities).idxmax(axis=1))
        cluster_labels_array=np.unique(np.array(pd.DataFrame(posterior_probabilities).idxmax(axis=1)))
        list_of_clusters=np.array([i for i in range(0,no_of_clusters)])
        missing_clusters=set(list_of_clusters)-set(cluster_labels_array)
        for missing_value in missing_clusters:
            unique_values,value_counts=np.unique(cluster_labels_original,return_counts=True)
            values_to_replace=unique_values[value_counts > 1]
            value_to_replace=np.random.choice(values_to_replace)
            indices=np.where(cluster_labels_original==value_to_replace)[0]
            random_index=np.random.choice(indices)
            new_value=missing_value
            cluster_labels_original[random_index]=new_value
        cluster_labels_array=cluster_labels_original
        within_sum_of_square_error=wcss_emm(df_diabetes_scaled,cluster_labels_array,no_of_clusters)
        #silheoutte_score_value=silheoutte_score(df_diabetes_scaled,cluster_labels_array)
        #print("C2")
        Calinski_Harbaz_score_value=Calinski_Harbaz_score(df_diabetes_scaled,cluster_labels_array)
        dbs_value=davies_bouldin_score(df_diabetes_scaled,cluster_labels_array)
        expectation_maximization_statistics.append([no_of_clusters,no_of_experiments,within_sum_of_square_error,silheoutte_score_value,Calinski_Harbaz_score_value,dbs_value])
        print("Appended_to_dataframe")
expectation_maximization_statistics_df= pd.DataFrame(expectation_maximization_statistics,columns=['No_of_Clusters', 'Iteration Number', 'within_sum_of_square_error','silheoutte_score','Calinski_Harbaz_score','davies_bouldin_score'])


# In[400]:


expectation_maximization_statistics_df_plot=expectation_maximization_statistics_df.groupby(['No_of_Clusters']).mean().reset_index()[['No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']]
expectation_maximization_statistics_df_plot


# In[404]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(5,5))
sns.boxplot(x=expectation_maximization_statistics_df['No_of_Clusters'],y=expectation_maximization_statistics_df['within_sum_of_square_error'])
plt.title('Box Plot for GMM SSE (error vs no of clusters)')
plt.show()
import seaborn as sns
plt.figure(figsize=(5,5))
sns.boxplot(x=expectation_maximization_statistics_df['No_of_Clusters'],y=expectation_maximization_statistics_df['Calinski_Harbaz_score'])
plt.title('Box Plot for GMM Calinski Harabaz Score (Score vs no of clusters)')
plt.show()
import seaborn as sns
plt.figure(figsize=(5,5))
sns.boxplot(x=expectation_maximization_statistics_df['No_of_Clusters'],y=expectation_maximization_statistics_df['davies_bouldin_score'])
plt.title('Box Plot for GMM Davies Bouldin Score (Score vs no of clusters)')
plt.show()


# In[402]:


ax = expectation_maximization_statistics_df_plot.plot(x='No_of_Clusters', y='davies_bouldin_score')
ax2=expectation_maximization_statistics_df_plot.plot(x='No_of_Clusters', y='Calinski_Harbaz_score',secondary_y=True, ax=ax)
ax.set_xlabel('No_of_Clusters')
ax.set_ylabel('Davies Bouldin Score')
ax2.set_ylabel('Calinski_Harabaz_Score')
ax.set_title('Cluster Indices vs No of Clusters')
ax.legend(['DBS'], loc='upper left')
ax2.legend(['CHS'], loc='upper right')
plt.show()


# # Initializing with K means ++

# In[768]:


from scipy.stats import multivariate_normal
import numpy as np

def kmeans_pp_init(input_dataframe,no_of_clusters):
    from scipy.spatial.distance import cdist
    '''
    K-means++ is a variant of the K-means algorithm that aims to improve the initial centroids' selection 
    in the clustering process. 
    The standard K-means algorithm initializes the cluster centroids randomly, 
    which can lead to suboptimal clustering results, 
    especially if the dataset has complex or irregular structures.
    '''
    list_of_centroids=[]
    #Choosing the first centroid randomly
    centroid = input_dataframe.apply(lambda x: float(x.sample()))
    list_of_centroids.append(centroid)
    
    iterator=2
    while iterator<=no_of_clusters:
        '''
        Calculating the distances from the centroid to every data point
        If the no of centroids are more than 1 calculate the distance from every centroid and take minimum distance
        '''
        distances = np.array(np.amin(cdist(input_dataframe,list_of_centroids,metric='euclidean'),axis=1))
        #Next centroid will be selected with probability proportional to the distance
        
        probs = distances / np.sum(distances)
        '''
        Selection of the next centroids
        '''
        next_centroid = input_dataframe.iloc[np.random.choice(len(input_dataframe),p=probs)]
        list_of_centroids.append(next_centroid)
        iterator+=1
    
    centroid_df=pd.concat(list_of_centroids,axis=1,ignore_index=True)
    return centroid_df.T

def initialization_of_GMM_Kmeans(input_dataframe,no_of_clusters):
    '''
    The function takes scaled dataframe as input and initializes the GMM means,Covariances,and Weights
    '''
    input_dataframe_values = input_dataframe.values
    row, column = input_dataframe_values.shape
    # Randomly initialize means vector
    means_vector = np.array(kmeans_pp_init(input_dataframe,no_of_clusters))
    # Initialize covariance matrices for each cluster
    covariances_vector = np.array([np.eye(column)] * no_of_clusters)
    # Initialize weights from uniform distribution
    weights_vector = np.ones(no_of_clusters) / no_of_clusters
    return means_vector,covariances_vector,weights_vector
    
    
def fit_Guassian_mixture_models_kmeans_plus_plus(input_dataframe,no_of_clusters,max_no_of_iterations,threshold):
    input_dataframe_values = input_dataframe.values
    row, column = input_dataframe_values.shape
    means,covariances,weights=initialization_of_GMM_Kmeans(input_dataframe,no_of_clusters)
    iteration = 0
    previous_log_likelihood_scalar=0
    while iteration < max_no_of_iterations:
       
        new_log_likelihood = 0
        for index in range(no_of_clusters):
            try:
                epsilon_weight=1e-6
                cov_inv = np.linalg.pinv(covariances[index] + np.diag(np.ones(covariances[index].shape[0]) * epsilon_weight))
                new_log_likelihood=new_log_likelihood+weights[index]*multivariate_normal.logpdf(input_dataframe_values,means[index], cov_inv)
            except np.linalg.LinAlgError as e:
                continue
        new_log_likelihood_scalar=np.sum(new_log_likelihood)
        
        '''
        Calculating percentage change
        '''
        if np.abs(((np.abs(new_log_likelihood_scalar-previous_log_likelihood_scalar)/new_log_likelihood_scalar)*100))<threshold:
            print("The input Threshold was {}".format(threshold))
            print("The calculated threshold is {}".format(np.abs(((np.abs(new_log_likelihood_scalar-previous_log_likelihood_scalar)/new_log_likelihood_scalar)*100))))
            break
        #else:
            #print("The input Threshold was {}".format(threshold))
            #print("The calculated threshold is {}".format(np.abs(((np.abs(new_log_likelihood_scalar-previous_log_likelihood_scalar)/new_log_likelihood_scalar)*100))))
            
        
        previous_log_likelihood_scalar=new_log_likelihood_scalar
        
        posterior_probabilities = np.zeros((len(input_dataframe_values),no_of_clusters))
        for index in range(no_of_clusters):
            try:
                cov_inv = np.linalg.pinv(covariances[index],rcond=1e-10)
            except np.linalg.LinAlgError as e:
                continue
            try:
                posterior_probabilities[:,index] = weights[index] * multivariate_normal.pdf(input_dataframe_values, means[index], cov_inv)
            except np.linalg.LinAlgError as e:
                continue
        posterior_probabilities/=np.sum(posterior_probabilities, axis=1, keepdims=True)

        
        
        for j in range(no_of_clusters):
            weighted_sum = np.zeros((1, means.shape[1]))
            sum_posterior = 0.0
            for i in range(row):
                weighted_sum += posterior_probabilities[i][j] * input_dataframe_values[i]
                sum_posterior += posterior_probabilities[i][j]
            means[j] = weighted_sum/sum_posterior
            difference = input_dataframe_values - means[j]
            covariances[j] = np.dot((difference * posterior_probabilities[:, j][:, np.newaxis]).T, difference) / np.sum(posterior_probabilities[:, j])
            covariances[j] += np.diag(np.ones(column) * 1e-6)
            weights[j] = np.mean(posterior_probabilities[:, j])
        
        
        
        iteration += 1
        
    return means,posterior_probabilities


# In[770]:


expectation_maximization_statistics_kmeans_plus_plus=[]
for no_of_clusters in range(2,6):
    print(no_of_clusters)
    for no_of_experiments in range(1,21):
        print(no_of_experiments)
        means,posterior_probabilities=fit_Guassian_mixture_models_kmeans_plus_plus(df_diabetes_scaled,no_of_clusters,100,1)
        cluster_labels_original=np.array(pd.DataFrame(posterior_probabilities).idxmax(axis=1))
        cluster_labels_array=np.unique(np.array(pd.DataFrame(posterior_probabilities).idxmax(axis=1)))
        list_of_clusters=np.array([i for i in range(0,no_of_clusters)])
        missing_clusters=set(list_of_clusters)-set(cluster_labels_array)
        for missing_value in missing_clusters:
            unique_values,value_counts=np.unique(cluster_labels_original,return_counts=True)
            values_to_replace=unique_values[value_counts > 1]
            value_to_replace=np.random.choice(values_to_replace)
            indices=np.where(cluster_labels_original==value_to_replace)[0]
            random_index=np.random.choice(indices)
            new_value=missing_value
            cluster_labels_original[random_index]=new_value
        cluster_labels_array=cluster_labels_original
        within_sum_of_square_error=wcss_emm(df_diabetes_scaled,cluster_labels_array,no_of_clusters)
        #silheoutte_score_value=silheoutte_score(df_diabetes_scaled,cluster_labels_array)
        #print("C2")
        Calinski_Harbaz_score_value=Calinski_Harbaz_score(df_diabetes_scaled,cluster_labels_array)
        dbs_value=davies_bouldin_score(df_diabetes_scaled,cluster_labels_array)
        expectation_maximization_statistics_kmeans_plus_plus.append([no_of_clusters,no_of_experiments,within_sum_of_square_error,silheoutte_score_value,Calinski_Harbaz_score_value,dbs_value])
        print("Appended_to_dataframe")
expectation_maximization_statistics_kmeans_plus_plus_df= pd.DataFrame(expectation_maximization_statistics_kmeans_plus_plus,columns=['No_of_Clusters', 'Iteration Number', 'within_sum_of_square_error','silheoutte_score','Calinski_Harbaz_score','davies_bouldin_score'])


# In[396]:


expectation_maximization_statistics_kmeans_plus_plus_df_plot=expectation_maximization_statistics_kmeans_plus_plus_df.groupby(['No_of_Clusters']).mean().reset_index()[['No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']]


# # Comparing GMM and GMM++

# In[446]:


import seaborn as sns
expectation_maximization_statistics_kmeans_plus_plus_df['algorithm']='GMM_K++'
expectation_maximization_statistics_df['algorithm']='GMM'
comparison_df=pd.DataFrame()
comparison_df=pd.concat([expectation_maximization_statistics_kmeans_plus_plus_df[['algorithm','No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']],
expectation_maximization_statistics_df[['algorithm','No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']]
],ignore_index=True )
fig, ax = plt.subplots(figsize=(5,5))
sns.boxplot(x='No_of_Clusters', y='Calinski_Harbaz_score', hue='algorithm',
data=comparison_df[comparison_df['algorithm'].isin (['GMM_K++','GMM'])],ax=ax);
plt.title('Box Plot of Calinski Harbaz Score for GMM and GMM_K++')
plt.show()


# In[444]:


import seaborn as sns
expectation_maximization_statistics_kmeans_plus_plus_df['algorithm']='GMM_K++'
expectation_maximization_statistics_df['algorithm']='GMM'
comparison_df=pd.DataFrame()
comparison_df=pd.concat([expectation_maximization_statistics_kmeans_plus_plus_df[['algorithm','No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']],
expectation_maximization_statistics_df[['algorithm','No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']]
],ignore_index=True )
fig, ax = plt.subplots(figsize=(5,5))
sns.boxplot(x='No_of_Clusters', y='davies_bouldin_score', hue='algorithm',
data=comparison_df[comparison_df['algorithm'].isin (['GMM_K++','GMM'])],ax=ax);
plt.title('Box Plot of Davies Bouldin Score for GMM and GMM_K++')
plt.show()


# In[ ]:


import seaborn as sns
expectation_maximization_statistics_kmeans_plus_plus_df['algorithm']='GMM_K++'
expectation_maximization_statistics_df['algorithm']='GMM'
comparison_df=pd.DataFrame()
comparison_df=pd.concat([expectation_maximization_statistics_kmeans_plus_plus_df[['algorithm','No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']],
expectation_maximization_statistics_df[['algorithm','No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']]
],ignore_index=True )
fig, ax = plt.subplots(figsize=(5,5))
sns.boxplot(x='No_of_Clusters', y='within_sum_of_square_error', hue='algorithm',
data=comparison_df[comparison_df['algorithm'].isin (['GMM_K++','GMM'])],ax=ax);
plt.title('Box Plot of SSE for GMM and GMM_K++')
plt.show()


# # Running k means ++ on dataset

# In[454]:


df_diabetes_scaled.columns=df_diabetes.columns


# In[455]:


error_values_kmeans_plus_plus_alone=[]
for no_of_clusters in range(2,6):
    print(no_of_clusters)
    for no_of_experiments in range(1,21):
        print(no_of_experiments)
        final_centroids,sum_of_squared_error,sil_score,chs_score,dbs_score,run_time,same_centroid=kmeans_plus_plus(df_diabetes_scaled,no_of_clusters,10,100)
        error_values_kmeans_plus_plus_alone.append([no_of_clusters,no_of_experiments,sum_of_squared_error,sil_score,chs_score,dbs_score,run_time])
error_values_kmeans_plus_plus_alone_df= pd.DataFrame(error_values_kmeans_plus_plus_alone,columns=['No_of_Clusters', 'Iteration Number','within_sum_of_square_error','Silheoutte_Score','Calinski_Harbaz_score','davies_bouldin_score','run_time'])  


# In[458]:


error_values_kmeans_plus_plus_alone_df.head(1)


# # Comparing K means ++ and GMM++

# In[457]:


import seaborn as sns
expectation_maximization_statistics_kmeans_plus_plus_df['algorithm']='GMM_K++'
error_values_kmeans_plus_plus_alone_df['algorithm']='K++'
comparison_df=pd.DataFrame()
comparison_df=pd.concat([expectation_maximization_statistics_kmeans_plus_plus_df[['algorithm','No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']],
error_values_kmeans_plus_plus_alone_df[['algorithm','No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']]
],ignore_index=True )
fig, ax = plt.subplots(figsize=(5,5))
sns.boxplot(x='No_of_Clusters', y='Calinski_Harbaz_score', hue='algorithm',
data=comparison_df[comparison_df['algorithm'].isin (['GMM_K++','K++'])],ax=ax);
plt.title('Box Plot of Calinski Harbaz Score for K++ and GMM_K++')
plt.show()


# In[459]:


import seaborn as sns
expectation_maximization_statistics_kmeans_plus_plus_df['algorithm']='GMM_K++'
error_values_kmeans_plus_plus_alone_df['algorithm']='K++'
comparison_df=pd.DataFrame()
comparison_df=pd.concat([expectation_maximization_statistics_kmeans_plus_plus_df[['algorithm','No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']],
error_values_kmeans_plus_plus_alone_df[['algorithm','No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']]
],ignore_index=True )
fig, ax = plt.subplots(figsize=(5,5))
sns.boxplot(x='No_of_Clusters', y='within_sum_of_square_error', hue='algorithm',
data=comparison_df[comparison_df['algorithm'].isin (['GMM_K++','K++'])],ax=ax);
plt.title('Box Plot of Within Sum of Square Error for K++ and GMM_K++')
plt.show()


# In[460]:


import seaborn as sns
expectation_maximization_statistics_kmeans_plus_plus_df['algorithm']='GMM_K++'
error_values_kmeans_plus_plus_alone_df['algorithm']='K++'
comparison_df=pd.DataFrame()
comparison_df=pd.concat([expectation_maximization_statistics_kmeans_plus_plus_df[['algorithm','No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']],
error_values_kmeans_plus_plus_alone_df[['algorithm','No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']]
],ignore_index=True )
fig, ax = plt.subplots(figsize=(5,5))
sns.boxplot(x='No_of_Clusters', y='davies_bouldin_score', hue='algorithm',
data=comparison_df[comparison_df['algorithm'].isin (['GMM_K++','K++'])],ax=ax);
plt.title('Box Plot of davies bouldin score for K++ and GMM_K++')
plt.show()


# # Running K means on dataset without K++

# In[463]:


import numpy as np
import swifter
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist
import time
def get_random_centroids(input_dataframe,no_of_clusters):
    list_of_centroids = []
    
    for cluster in range(no_of_clusters):
        random_centroid = input_dataframe.swifter.apply(lambda x:float(x.sample()))
        list_of_centroids.append(random_centroid)
    
    centroid_df=pd.concat(list_of_centroids,axis=1)
    
    centroid_df.index.name='Cluster_Assigned'
    
    return centroid_df


def get_labels(input_dataframe,centroid_df):
    euclidean_distances = centroid_df.swifter.apply(lambda x: np.sqrt(((input_dataframe - x) ** 2).sum(axis=1)))
    return pd.DataFrame(euclidean_distances.idxmin(axis=1))

        
def get_new_centroids(df_clustered_label,input_dataframe):
    df_original_label_join=input_dataframe.join(df_clustered_label)
    df_original_label_join.rename(columns={0:'Cluster_Assigned'},inplace=True)
    new_centroids=df_original_label_join.groupby('Cluster_Assigned').mean()
    return new_centroids.T


def kmeans_llyod(input_dataframe,no_of_clusters,threshold,no_of_iterations):
    start_time=time.time()
    iteration=0
    initial_centroid=get_random_centroids(input_dataframe,no_of_clusters)
    same_centroid=initial_centroid
    initial_centroid_column_list=initial_centroid.columns.to_list()
    
    while True:
        
        df_cluster_label=get_labels(input_dataframe,initial_centroid)
        df_new_centroids=get_new_centroids(df_cluster_label,input_dataframe)
        new_list_of_columns=df_new_centroids.columns.to_list()
        initial_set_columns = set(initial_centroid_column_list)
        new_set_columns = set(new_list_of_columns)
        missing_columns = initial_set_columns - new_set_columns
        for col in missing_columns:
            df_new_centroids[col]=initial_centroid[col]
        
        from scipy.spatial.distance import euclidean
        scalar_product = [euclidean(initial_centroid[col],df_new_centroids[col]) for col in initial_centroid.columns]
        threshold_calculated=float(sum(scalar_product))/no_of_clusters
        
        iteration+=1
        
        if threshold_calculated<threshold:
            print("The input Threshold was {}".format(threshold))
            print("The calculated threshold is {}".format(threshold_calculated))
        
        if iteration>no_of_iterations:
            print("Limit for iterations has exceeded")
        
        if threshold_calculated<threshold or iteration>no_of_iterations:
            sum_of_square_error=sum_of_square_error_function(df_cluster_label,input_dataframe,df_new_centroids,no_of_clusters)
            df_cluster_label_copy=df_cluster_label.copy()
            df_cluster_label_copy.rename(columns={0:'Cluster_Assigned'},inplace=True)
            labels=df_cluster_label_copy['Cluster_Assigned'].to_list()
            #silheoutte_score=silheoutte_score_Kmeans(input_dataframe,labels)
            silheoutte_score=0
            chs_score=Calinski_Harbaz_score_Kmeans(input_dataframe,labels)
            dbs_score=davies_bouldin_score(input_dataframe,labels)
            end_time=time.time()
            return df_new_centroids,sum_of_square_error,silheoutte_score,chs_score,dbs_score,end_time-start_time,same_centroid
            break
        else:
            initial_centroid= df_new_centroids
        

def sum_of_square_error_function(df_cluster_label,input_dataframe,df_new_centroids,no_of_clusters):
    df_data_label=input_dataframe.join(df_cluster_label)
    df_data_label.rename(columns={0:'Cluster_Assigned'},inplace=True)
    total_error=[]
    for cluster in range(no_of_clusters):
        df_data_label_cluster=df_data_label[df_data_label['Cluster_Assigned']==cluster]
        df_data_label_cluster=df_data_label_cluster.drop('Cluster_Assigned',axis=1)
        centroids=pd.DataFrame(df_new_centroids[cluster])
        euclidean_distance=cdist(df_data_label_cluster,centroids.T,metric='euclidean')
        total_error.append(sum(euclidean_distance))
    return round(float(''.join(map(str, sum(total_error)))),3)

def silheoutte_score_Kmeans(input_dataframe,labels):
    from sklearn.metrics import silhouette_score
    silhouette_avg = silhouette_score(input_dataframe, labels)
    return silhouette_avg

def Calinski_Harbaz_score_Kmeans(input_dataframe,labels):
    from sklearn.metrics import calinski_harabasz_score
    chs=calinski_harabasz_score(input_dataframe,labels)
    return chs


def davies_bouldin_score(input_dataframe,labels):
    from sklearn.metrics import davies_bouldin_score
    dbs=davies_bouldin_score(input_dataframe,labels)
    return dbs


# In[466]:


error_values_kmeans_alone=[]
for no_of_clusters in range(2,6):
    print(no_of_clusters)
    for no_of_experiments in range(1,21):
        print(no_of_experiments)
        final_centroids,sum_of_squared_error,sil_score,chs_score,dbs_score,run_time,same_centroid=kmeans_llyod(df_diabetes_scaled,no_of_clusters,10,100)
        error_values_kmeans_alone.append([no_of_clusters,no_of_experiments,sum_of_squared_error,sil_score,chs_score,dbs_score,run_time])
error_values_kmeans_alone_df= pd.DataFrame(error_values_kmeans_alone,columns=['No_of_Clusters', 'Iteration Number','within_sum_of_square_error','Silheoutte_Score','Calinski_Harbaz_score','davies_bouldin_score','run_time'])  


# # Comparing everything K means,K means ++,EM,EM++

# In[468]:


import seaborn as sns
expectation_maximization_statistics_kmeans_plus_plus_df['algorithm']='GMM_K++'
error_values_kmeans_plus_plus_alone_df['algorithm']='K++'
expectation_maximization_statistics_df['algorithm']='GMM'
error_values_kmeans_alone_df['algorithm']='K_Means'
comparison_df=pd.DataFrame()
comparison_df=pd.concat([expectation_maximization_statistics_kmeans_plus_plus_df[['algorithm','No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']],
error_values_kmeans_plus_plus_alone_df[['algorithm','No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']],
expectation_maximization_statistics_df[['algorithm','No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']],
error_values_kmeans_alone_df[['algorithm','No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']]                         
],ignore_index=True)
fig, ax = plt.subplots(figsize=(10,10))
sns.boxplot(x='No_of_Clusters', y='davies_bouldin_score', hue='algorithm',
data=comparison_df[comparison_df['algorithm'].isin (['GMM_K++','K++','GMM','K_Means'])],ax=ax);
plt.title('Box Plot of davies bouldin score for K++ and GMM_K++')
plt.show()


# In[470]:


import seaborn as sns
expectation_maximization_statistics_kmeans_plus_plus_df['algorithm']='GMM_K++'
error_values_kmeans_plus_plus_alone_df['algorithm']='K++'
expectation_maximization_statistics_df['algorithm']='GMM'
error_values_kmeans_alone_df['algorithm']='K_Means'
comparison_df=pd.DataFrame()
comparison_df=pd.concat([expectation_maximization_statistics_kmeans_plus_plus_df[['algorithm','No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']],
error_values_kmeans_plus_plus_alone_df[['algorithm','No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']],
expectation_maximization_statistics_df[['algorithm','No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']],
error_values_kmeans_alone_df[['algorithm','No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']]                         
],ignore_index=True)
fig, ax = plt.subplots(figsize=(10,10))
sns.boxplot(x='No_of_Clusters', y='within_sum_of_square_error', hue='algorithm',
data=comparison_df[comparison_df['algorithm'].isin (['GMM_K++','K++','GMM','K_Means'])],ax=ax);
plt.title('Box Plot of Within Sum of Square Error for K++ and GMM_K++')
plt.show()


# In[471]:


import seaborn as sns
expectation_maximization_statistics_kmeans_plus_plus_df['algorithm']='GMM_K++'
error_values_kmeans_plus_plus_alone_df['algorithm']='K++'
expectation_maximization_statistics_df['algorithm']='GMM'
error_values_kmeans_alone_df['algorithm']='K_Means'
comparison_df=pd.DataFrame()
comparison_df=pd.concat([expectation_maximization_statistics_kmeans_plus_plus_df[['algorithm','No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']],
error_values_kmeans_plus_plus_alone_df[['algorithm','No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']],
expectation_maximization_statistics_df[['algorithm','No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']],
error_values_kmeans_alone_df[['algorithm','No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']]                         
],ignore_index=True)
fig, ax = plt.subplots(figsize=(10,10))
sns.boxplot(x='No_of_Clusters', y='Calinski_Harbaz_score', hue='algorithm',
data=comparison_df[comparison_df['algorithm'].isin (['GMM_K++','K++','GMM','K_Means'])],ax=ax);
plt.title('Box Plot of Calinski_Harbaz_score for K++ and GMM_K++')
plt.show()


# # Performing PCA on data

# In[417]:


import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

def covariance(input_dataframe):
    '''
    This function takse input as a standardized dataframe
    '''
    input_dataframe_mean = input_dataframe.swifter.apply(np.mean, axis=0)
    input_dataframe_centered= input_dataframe-input_dataframe_mean
    with tqdm(total=input_dataframe.shape[1], desc="Calculating Covariance Matrix") as pbar:
        cov_matrix=np.cov(input_dataframe.T)
        pbar.update()
    return cov_matrix,input_dataframe_centered

def principal_component_analysis(input_dataframe):
    '''
    This function takes input_dataframe,stadndardizes it and number of components as the number of components required by PC
    '''
    scaler = StandardScaler()
    input_dataframe_scaled =pd.DataFrame(scaler.fit_transform(input_dataframe))
    #Calling the covriance function
    covariance_matrix,input_dataframe_centered=covariance(input_dataframe_scaled)
    #Calculates Covariance Matirx
    eigen_values,eigen_vectors=np.linalg.eig(covariance_matrix)
    #Calculates Eigen Values and Eigen Vectors
    sorted_indices=np.argsort(eigen_values)
    #Sort the elements in descending order
    sorted_indices=sorted_indices[::-1]
    
    
    explained_variances = eigen_values / np.sum(eigen_values)
    
    variance_explained_ratios = pd.DataFrame(explained_variances[sorted_indices], columns=["variance_explained_ratio"])
    variance_explained_ratios["cumulative_variance_explained_ratio"] = variance_explained_ratios["variance_explained_ratio"].cumsum()
    
    #Find the number of components that explain 90% of variance
    number_of_components = variance_explained_ratios["cumulative_variance_explained_ratio"][variance_explained_ratios["cumulative_variance_explained_ratio"] <= 0.90].count() + 1
    
    print("Number of Principal components explain 90% of variance are {}".format(number_of_components))
    
    
    
    
    #Taking Top Eigen Values and Top Eigen Vectors
    top_eigen_values_indices=sorted_indices[:number_of_components]
    top_eigen_vectors=eigen_vectors[:,top_eigen_values_indices]
    
     #Variance Calculations Plot
    explained_variances = eigen_values/np.sum(eigen_values)
    variance_explained = pd.DataFrame(eigen_values[top_eigen_values_indices] / sum(eigen_values))
    variance_explained['PC_Feature']=top_eigen_values_indices
    variance_explained_plot=pd.Series(eigen_values[top_eigen_values_indices] / sum(eigen_values))
    
    
    #Cumulative Variance Plot
    cumulative_variance_explained = np.cumsum(variance_explained_plot)
    cumulative_variance_explained_plot = pd.Series(cumulative_variance_explained)
    
    
    
    #Projecting Principal Components 
    principal_components=input_dataframe_centered.dot(top_eigen_vectors)
    principal_components.columns=[f'PC{i+1}' for i in range(number_of_components)]
    
    
   
    
    #Calculate the loadings
    loadings = pd.DataFrame(top_eigen_vectors,index=input_dataframe.columns)
    
    df_principal_components=pd.DataFrame(principal_components, columns=[f'PC{i+1}' for i in range(number_of_components)])
    #PLotting the graph 
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(np.arange(1, number_of_components+1),variance_explained_plot, 'o-')
    ax[0].set_xlabel('Principal Component')
    ax[0].set_ylabel('Proportion of Variance Explained')
    ax[0].set_title('Scree Plot')
    
    
    ax[1].plot(np.arange(1, number_of_components+1),cumulative_variance_explained_plot, 'o-')
    ax[1].set_xlabel('Principal Component')
    ax[1].set_ylabel('Cumulative Proportion of Variance Explained')
    ax[1].set_title('Cumulative Scree Plot')
    plt.tight_layout()
    plt.show()
    
    #Correlation between PC1 and PC2
    
    plt.scatter(principal_components['PC1'], principal_components['PC2'])
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Scatter plot of PC1 against PC2')
    plt.show()
    
    principal_components_temp=principal_components[['PC1','PC2']]
    corr_matrix = principal_components_temp.corr()
    print('Correlation matrix:')
    print(corr_matrix)
    
    total_variance_explained=cumulative_variance_explained_plot[1]
    print("The total variance explained by first two PC's is {}".format(total_variance_explained))

    return variance_explained,loadings,principal_components,cumulative_variance_explained
    


# In[418]:


variance_explained,loadings,principal_components,cumulative_variance_explained=principal_component_analysis(df_diabetes_scaled)


# In[ ]:





# # Running EM Algorithm on reduced dataset with K means plus plus init

# In[801]:


expectation_maximization_statistics_kmeans_plus_plus_pca=[]
for no_of_clusters in range(2,6):
    print(no_of_clusters)
    for no_of_experiments in range(1,21):
        print(no_of_experiments)
        means,posterior_probabilities=fit_Guassian_mixture_models_kmeans_plus_plus(principal_components,no_of_clusters,100,1)
        cluster_labels_original=np.array(pd.DataFrame(posterior_probabilities).idxmax(axis=1))
        cluster_labels_array=np.unique(np.array(pd.DataFrame(posterior_probabilities).idxmax(axis=1)))
        list_of_clusters=np.array([i for i in range(0,no_of_clusters)])
        missing_clusters=set(list_of_clusters)-set(cluster_labels_array)
        for missing_value in missing_clusters:
            unique_values,value_counts=np.unique(cluster_labels_original,return_counts=True)
            values_to_replace=unique_values[value_counts > 1]
            value_to_replace=np.random.choice(values_to_replace)
            indices=np.where(cluster_labels_original==value_to_replace)[0]
            if len(indices)==0:
                indices=[0]
            random_index=np.random.choice(indices)
            new_value=missing_value
            cluster_labels_original[random_index]=new_value
        cluster_labels_array=cluster_labels_original
        try:
            within_sum_of_square_error=wcss_emm(principal_components,cluster_labels_array,no_of_clusters)
        except KeyError as e:
            continue
        Calinski_Harbaz_score_value=Calinski_Harbaz_score(principal_components,cluster_labels_array)
        dbs_value=davies_bouldin_score(principal_components,cluster_labels_array)
        expectation_maximization_statistics_kmeans_plus_plus_pca.append([no_of_clusters,no_of_experiments,within_sum_of_square_error,silheoutte_score_value,Calinski_Harbaz_score_value,dbs_value])
        print("Appended_to_dataframe")
expectation_maximization_statistics_kmeans_plus_plus_pca_df= pd.DataFrame(expectation_maximization_statistics_kmeans_plus_plus_pca,columns=['No_of_Clusters', 'Iteration Number', 'within_sum_of_square_error','silheoutte_score','Calinski_Harbaz_score','davies_bouldin_score'])


# In[440]:


expectation_maximization_statistics_kmeans_plus_plus_pca_df


# In[810]:


expectation_maximization_statistics_kmeans_plus_plus_pca_df_plot


# # Running K means on Reduced Dataset

# In[423]:


import numpy as np
import swifter
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist
import time
def kmeans_pp_init(input_dataframe,no_of_clusters):
    '''
    K-means++ is a variant of the K-means algorithm that aims to improve the initial centroids' selection 
    in the clustering process. 
    The standard K-means algorithm initializes the cluster centroids randomly, 
    which can lead to suboptimal clustering results, 
    especially if the dataset has complex or irregular structures.
    '''
    list_of_centroids=[]
    #Choosing the first centroid randomly
    centroid = input_dataframe.apply(lambda x: float(x.sample()))
    list_of_centroids.append(centroid)
    
    iterator=2
    while iterator<=no_of_clusters:
        '''
        Calculating the distances from the centroid to every data point
        If the no of centroids are more than 1 calculate the distance from every centroid and take minimum distance
        '''
        distances = np.array(np.amin(cdist(input_dataframe,list_of_centroids,metric='euclidean'),axis=1))
        #Next centroid will be selected with probability proportional to the distance
        
        probs = distances / np.sum(distances)
        '''
        Selection of the next centroids
        '''
        next_centroid = input_dataframe.iloc[np.random.choice(len(input_dataframe),p=probs)]
        list_of_centroids.append(next_centroid)
        iterator+=1
    
    centroid_df=pd.concat(list_of_centroids,axis=1,ignore_index=True)
    #Naming the column as Label for ease of purpose
    centroid_df.index.name='Cluster_Assigned'   
    
        
    return centroid_df


def get_labels(input_dataframe,centroid_df):
    euclidean_distances = centroid_df.swifter.apply(lambda x: np.sqrt(((input_dataframe - x) ** 2).sum(axis=1)))
    return pd.DataFrame(euclidean_distances.idxmin(axis=1))

        
def get_new_centroids(df_clustered_label,input_dataframe):
    df_original_label_join=input_dataframe.join(df_clustered_label)
    df_original_label_join.rename(columns={0:'Cluster_Assigned'},inplace=True)
    new_centroids=df_original_label_join.groupby('Cluster_Assigned').mean()
    return new_centroids.T


def kmeans_plus_plus(input_dataframe,no_of_clusters,threshold,no_of_iterations):
    start_time=time.time()
    iteration=0
    initial_centroid=kmeans_pp_init(input_dataframe,no_of_clusters)
    same_centroid=initial_centroid
    initial_centroid_column_list=initial_centroid.columns.to_list()
    
    while True:
        
        df_cluster_label=get_labels(input_dataframe,initial_centroid)
        df_new_centroids=get_new_centroids(df_cluster_label,input_dataframe)
        new_list_of_columns=df_new_centroids.columns.to_list()
        initial_set_columns = set(initial_centroid_column_list)
        new_set_columns = set(new_list_of_columns)
        missing_columns = initial_set_columns - new_set_columns
        for col in missing_columns:
            df_new_centroids[col]=initial_centroid[col]
        
        from scipy.spatial.distance import euclidean
        scalar_product = [euclidean(initial_centroid[col],df_new_centroids[col]) for col in initial_centroid.columns]
        threshold_calculated=float(sum(scalar_product))/no_of_clusters
        
        iteration+=1
        
        if threshold_calculated<threshold:
            print("The input Threshold was {}".format(threshold))
            print("The calculated threshold is {}".format(threshold_calculated))
        
        if iteration>no_of_iterations:
            print("Limit for iterations has exceeded")
        
        if threshold_calculated<threshold or iteration>no_of_iterations:
            sum_of_square_error=sum_of_square_error_function(df_cluster_label,input_dataframe,df_new_centroids,no_of_clusters)
            df_cluster_label_copy=df_cluster_label.copy()
            df_cluster_label_copy.rename(columns={0:'Cluster_Assigned'},inplace=True)
            labels=df_cluster_label_copy['Cluster_Assigned'].to_list()
            #silheoutte_score=silheoutte_score_Kmeans(input_dataframe,labels)
            silheoutte_score=0
            chs_score=Calinski_Harbaz_score_Kmeans(input_dataframe,labels)
            dbs_score=davies_bouldin_score(input_dataframe,labels)
            end_time=time.time()
            return df_new_centroids,sum_of_square_error,silheoutte_score,chs_score,dbs_score,end_time-start_time,same_centroid
            break
        else:
            initial_centroid= df_new_centroids
        

def sum_of_square_error_function(df_cluster_label,input_dataframe,df_new_centroids,no_of_clusters):
    df_data_label=input_dataframe.join(df_cluster_label)
    df_data_label.rename(columns={0:'Cluster_Assigned'},inplace=True)
    total_error=[]
    for cluster in range(no_of_clusters):
        df_data_label_cluster=df_data_label[df_data_label['Cluster_Assigned']==cluster]
        df_data_label_cluster=df_data_label_cluster.drop('Cluster_Assigned',axis=1)
        centroids=pd.DataFrame(df_new_centroids[cluster])
        euclidean_distance=cdist(df_data_label_cluster,centroids.T,metric='euclidean')
        total_error.append(sum(euclidean_distance))
    return round(float(''.join(map(str, sum(total_error)))),3)

def silheoutte_score_Kmeans(input_dataframe,labels):
    from sklearn.metrics import silhouette_score
    silhouette_avg = silhouette_score(input_dataframe, labels)
    return silhouette_avg

def Calinski_Harbaz_score_Kmeans(input_dataframe,labels):
    from sklearn.metrics import calinski_harabasz_score
    chs=calinski_harabasz_score(input_dataframe,labels)
    return chs


def davies_bouldin_score(input_dataframe,labels):
    from sklearn.metrics import davies_bouldin_score
    dbs=davies_bouldin_score(input_dataframe,labels)
    return dbs



# In[425]:


error_values_kmeans_pca=[]
for no_of_clusters in range(2,6):
    print(no_of_clusters)
    for no_of_experiments in range(1,21):
        print(no_of_experiments)
        final_centroids,sum_of_squared_error,sil_score,chs_score,dbs_score,run_time,same_centroid=kmeans_plus_plus(pd.DataFrame(principal_components),no_of_clusters,10,100)
        error_values_kmeans_pca.append([no_of_clusters,no_of_experiments,sum_of_squared_error,sil_score,chs_score,dbs_score,run_time])
error_values_kmeans_pca_df= pd.DataFrame(error_values_kmeans_pca,columns=['No_of_Clusters', 'Iteration Number','within_sum_of_square_error','Silheoutte_Score','Calinski_Harbaz_score','davies_bouldin_score','run_time'])  


# In[426]:


import seaborn as sns
expectation_maximization_statistics_kmeans_plus_plus_pca_df['algorithm']='GMM_PCA_K++'
error_values_kmeans_pca_df['algorithm']='K-Means++'
comparison_df=pd.DataFrame()
comparison_df=pd.concat([expectation_maximization_statistics_kmeans_plus_plus_pca_df[['algorithm','No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']],
expectation_maximization_statistics_kmeans_plus_plus_df[['algorithm','No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']]
],ignore_index=True )
fig, ax = plt.subplots(figsize=(5,5))
sns.boxplot(x='No_of_Clusters', y='within_sum_of_square_error', hue='algorithm',
data=comparison_df[comparison_df['algorithm'].isin (['K-Means++','GMM_PCA_K++'])],ax=ax);
plt.title('Box Plot of SSE for GMM_PCA and K means')
plt.show()


# In[427]:


import seaborn as sns
expectation_maximization_statistics_kmeans_plus_plus_pca_df['algorithm']='GMM_PCA_K++'
error_values_kmeans_pca_df['algorithm']='K-Means++'
comparison_df=pd.DataFrame()
comparison_df=pd.concat([expectation_maximization_statistics_kmeans_plus_plus_pca_df[['algorithm','No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']],
expectation_maximization_statistics_kmeans_plus_plus_df[['algorithm','No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']]
],ignore_index=True )
fig, ax = plt.subplots(figsize=(5,5))
sns.boxplot(x='No_of_Clusters', y='davies_bouldin_score', hue='algorithm',
data=comparison_df[comparison_df['algorithm'].isin (['K-Means++','GMM_PCA_K++'])],ax=ax);
plt.title('Box Plot of Davies Bouldin Score for GMM_PCA and K means')
plt.show()


# In[431]:


import seaborn as sns
expectation_maximization_statistics_kmeans_plus_plus_pca_df['algorithm']='GMM_PCA_K++'
error_values_kmeans_pca_df['algorithm']='K-Means++'
comparison_df=pd.DataFrame()
comparison_df=pd.concat([expectation_maximization_statistics_kmeans_plus_plus_pca_df[['algorithm','No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']],
expectation_maximization_statistics_kmeans_plus_plus_df[['algorithm','No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']]
],ignore_index=True )
fig, ax = plt.subplots(figsize=(5,5))
sns.boxplot(x='No_of_Clusters', y='Calinski_Harbaz_score', hue='algorithm',
data=comparison_df[comparison_df['algorithm'].isin (['K-Means++','GMM_PCA_K++'])],ax=ax);
plt.title('Box Plot of Calinski Harbaz Score for GMM_PCA and K means')
plt.show()


# # K means with inital clusters and saving them for EMM

# In[841]:


import numpy as np
import swifter
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist
import time
def get_random_centroids(input_dataframe,no_of_clusters):
    list_of_centroids = []
    for cluster in range(no_of_clusters):
        random_centroid = input_dataframe.swifter.apply(lambda x:float(x.sample()))
        list_of_centroids.append(random_centroid)
    
    centroid_df=pd.concat(list_of_centroids,axis=1)
    centroid_df.index.name='Cluster_Assigned'
    return centroid_df

def get_labels(input_dataframe,centroid_df):
    euclidean_distances = centroid_df.swifter.apply(lambda x: np.sqrt(((input_dataframe - x) ** 2).sum(axis=1)))
    return pd.DataFrame(euclidean_distances.idxmin(axis=1))

        
def get_new_centroids(df_clustered_label,input_dataframe):
    df_original_label_join=input_dataframe.join(df_clustered_label)
    df_original_label_join.rename(columns={0:'Cluster_Assigned'},inplace=True)
    new_centroids=df_original_label_join.groupby('Cluster_Assigned').mean()
    return new_centroids.T


def kmeans_llyod(input_dataframe,no_of_clusters,threshold,no_of_iterations):
    start_time=time.time()
    iteration=0
    initial_centroid=get_random_centroids(input_dataframe,no_of_clusters)
    same_centroid=initial_centroid
    initial_centroid_column_list=initial_centroid.columns.to_list()
    
    while True:
        
        df_cluster_label=get_labels(input_dataframe,initial_centroid)
        df_new_centroids=get_new_centroids(df_cluster_label,input_dataframe)
        new_list_of_columns=df_new_centroids.columns.to_list()
        initial_set_columns = set(initial_centroid_column_list)
        new_set_columns = set(new_list_of_columns)
        missing_columns = initial_set_columns - new_set_columns
        for col in missing_columns:
            df_new_centroids[col]=initial_centroid[col]
        
        from scipy.spatial.distance import euclidean
        scalar_product = [euclidean(initial_centroid[col],df_new_centroids[col]) for col in initial_centroid.columns]
        threshold_calculated=float(sum(scalar_product))/no_of_clusters
        
        iteration+=1
        
        if threshold_calculated<threshold:
            print("The input Threshold was {}".format(threshold))
            print("The calculated threshold is {}".format(threshold_calculated))
        
        if iteration>no_of_iterations:
            print("Limit for iterations has exceeded")
        
        if threshold_calculated<threshold or iteration>no_of_iterations:
            sum_of_square_error=sum_of_square_error_function(df_cluster_label,input_dataframe,df_new_centroids,no_of_clusters)
            df_cluster_label_copy=df_cluster_label.copy()
            df_cluster_label_copy.rename(columns={0:'Cluster_Assigned'},inplace=True)
            labels=df_cluster_label_copy['Cluster_Assigned'].to_list()
            #silheoutte_score=silheoutte_score_Kmeans(input_dataframe,labels)
            silheoutte_score=0
            chs_score=Calinski_Harbaz_score_Kmeans(input_dataframe,labels)
            dbs_score=davies_bouldin_score(input_dataframe,labels)
            end_time=time.time()
            return df_new_centroids,sum_of_square_error,silheoutte_score,chs_score,dbs_score,end_time-start_time,same_centroid
            break
        else:
            initial_centroid= df_new_centroids
        

def sum_of_square_error_function(df_cluster_label,input_dataframe,df_new_centroids,no_of_clusters):
    df_data_label=input_dataframe.join(df_cluster_label)
    df_data_label.rename(columns={0:'Cluster_Assigned'},inplace=True)
    total_error=[]
    for cluster in range(no_of_clusters):
        df_data_label_cluster=df_data_label[df_data_label['Cluster_Assigned']==cluster]
        df_data_label_cluster=df_data_label_cluster.drop('Cluster_Assigned',axis=1)
        centroids=pd.DataFrame(df_new_centroids[cluster])
        euclidean_distance=cdist(df_data_label_cluster,centroids.T,metric='euclidean')
        total_error.append(sum(euclidean_distance))
    return round(float(''.join(map(str, sum(total_error)))),3)

def silheoutte_score_Kmeans(input_dataframe,labels):
    from sklearn.metrics import silhouette_score
    silhouette_avg = silhouette_score(input_dataframe, labels)
    return silhouette_avg

def Calinski_Harbaz_score_Kmeans(input_dataframe,labels):
    from sklearn.metrics import calinski_harabasz_score
    chs=calinski_harabasz_score(input_dataframe,labels)
    return chs


def davies_bouldin_score(input_dataframe,labels):
    from sklearn.metrics import davies_bouldin_score
    dbs=davies_bouldin_score(input_dataframe,labels)
    return dbs

from scipy.stats import multivariate_normal
import numpy as np

def initialization_of_GMM(input_dataframe,no_of_clusters):
    '''
    The function takes scaled dataframe as input and initializes the GMM means,Covariances,and Weights
    '''
    input_dataframe_values = input_dataframe.values
    row, column = input_dataframe_values.shape
    # Randomly initialize means vector
    means_vector = np.array(get_random_centroids(input_dataframe,no_of_clusters).T)
    # Initialize covariance matrices for each cluster
    covariances_vector = np.array([np.eye(column)] * no_of_clusters)
    # Initialize weights from uniform distribution
    weights_vector = np.ones(no_of_clusters)/no_of_clusters
    return means_vector,covariances_vector,weights_vector
    
    
def fit_Guassian_mixture_models(input_dataframe,no_of_clusters,max_no_of_iterations,threshold):
    input_dataframe_values = input_dataframe.values
    row, column = input_dataframe_values.shape
    means,covariances,weights=initialization_of_GMM(input_dataframe,no_of_clusters)
    iteration = 0
    previous_log_likelihood_scalar=0
    while iteration < max_no_of_iterations:
       
        new_log_likelihood = 0
        for index in range(no_of_clusters):
            try:
                epsilon_weight=1e-6
                cov_inv = np.linalg.pinv(covariances[index] + np.diag(np.ones(covariances[index].shape[0]) * epsilon_weight))
                new_log_likelihood=new_log_likelihood+weights[index]*multivariate_normal.logpdf(input_dataframe_values,means[index], cov_inv)
            except np.linalg.LinAlgError as e:
                continue
        new_log_likelihood_scalar=np.sum(new_log_likelihood)
        
        '''
        Calculating percentage change
        '''
        if np.abs(((np.abs(new_log_likelihood_scalar-previous_log_likelihood_scalar)/new_log_likelihood_scalar)*100))<threshold:
            print("The input Threshold was {}".format(threshold))
            print("The calculated threshold is {}".format(np.abs(((np.abs(new_log_likelihood_scalar-previous_log_likelihood_scalar)/new_log_likelihood_scalar)*100))))
            break
        #else:
            #print("The input Threshold was {}".format(threshold))
            #print("The calculated threshold is {}".format(np.abs(((np.abs(new_log_likelihood_scalar-previous_log_likelihood_scalar)/new_log_likelihood_scalar)*100))))
            
        
        previous_log_likelihood_scalar=new_log_likelihood_scalar
        
        posterior_probabilities = np.zeros((len(input_dataframe_values),no_of_clusters))
        for index in range(no_of_clusters):
            try:
                cov_inv = np.linalg.pinv(covariances[index],rcond=1e-10)
            except np.linalg.LinAlgError as e:
                continue
            try:
                posterior_probabilities[:,index] = weights[index] * multivariate_normal.pdf(input_dataframe_values, means[index], cov_inv)
            except np.linalg.LinAlgError as e:
                continue
        posterior_probabilities/=np.sum(posterior_probabilities, axis=1, keepdims=True)

        
        
        for j in range(no_of_clusters):
            weighted_sum = np.zeros((1, means.shape[1]))
            sum_posterior = 0.0
            for i in range(row):
                weighted_sum += posterior_probabilities[i][j] * input_dataframe_values[i]
                sum_posterior += posterior_probabilities[i][j]
            means[j] = weighted_sum/sum_posterior
            difference = input_dataframe_values - means[j]
            covariances[j] = np.dot((difference * posterior_probabilities[:, j][:, np.newaxis]).T, difference) / np.sum(posterior_probabilities[:, j])
            covariances[j] += np.diag(np.ones(column) * 1e-6)
            weights[j] = np.mean(posterior_probabilities[:, j])
        
        
        
        iteration += 1
        
    return means,posterior_probabilities


# In[842]:


error_values_kmeans_same_centroid=[]
error_values_emm_same_centroid=[]
for no_of_clusters in range(2,6):
    print(no_of_clusters)
    for no_of_experiments in range(1,21):
        print(no_of_experiments)
        final_centroids,sum_of_squared_error,sil_score,chs_score,dbs_score,run_time,same_centroid=kmeans_llyod(pd.DataFrame(df_diabetes_scaled),no_of_clusters,10,100)
        error_values_kmeans_same_centroid.append([no_of_clusters,no_of_experiments,sum_of_squared_error,sil_score,chs_score,dbs_score,run_time])
        means,posterior_probabilities=fit_Guassian_mixture_models(df_diabetes_scaled,no_of_clusters,100,1)
        cluster_labels_original=np.array(pd.DataFrame(posterior_probabilities).idxmax(axis=1))
        cluster_labels_array=np.unique(np.array(pd.DataFrame(posterior_probabilities).idxmax(axis=1)))
        list_of_clusters=np.array([i for i in range(0,no_of_clusters)])
        missing_clusters=set(list_of_clusters)-set(cluster_labels_array)
        for missing_value in missing_clusters:
            unique_values,value_counts=np.unique(cluster_labels_original,return_counts=True)
            values_to_replace=unique_values[value_counts > 1]
            value_to_replace=np.random.choice(values_to_replace)
            indices=np.where(cluster_labels_original==value_to_replace)[0]
            random_index=np.random.choice(indices)
            new_value=missing_value
            cluster_labels_original[random_index]=new_value
        cluster_labels_array=cluster_labels_original
        within_sum_of_square_error=wcss_emm(df_diabetes_scaled,cluster_labels_array,no_of_clusters)
        Calinski_Harbaz_score_value=Calinski_Harbaz_score(df_diabetes_scaled,cluster_labels_array)
        dbs_value=davies_bouldin_score(df_diabetes_scaled,cluster_labels_array)
        error_values_emm_same_centroid.append([no_of_clusters,no_of_experiments,within_sum_of_square_error,silheoutte_score_value,Calinski_Harbaz_score_value,dbs_value])
        print("Appended_to_dataframe")
expectation_maximization_statistics_same_centroid_df= pd.DataFrame(error_values_emm_same_centroid,columns=['No_of_Clusters', 'Iteration Number', 'within_sum_of_square_error','silheoutte_score','Calinski_Harbaz_score','davies_bouldin_score'])
error_values_kmeans_same_centroid_df= pd.DataFrame(error_values_kmeans_same_centroid,columns=['No_of_Clusters', 'Iteration Number','within_sum_of_square_error','Silheoutte_Score','Calinski_Harbaz_score','davies_bouldin_score','run_time'])  


# In[843]:


expectation_maximization_statistics_same_centroid_df


# In[844]:


error_values_kmeans_same_centroid_df


# # Saving all the Dataframes

# In[845]:


expectation_maximization_statistics_df.to_csv('expectation_maximization_statistics_df.csv')
expectation_maximization_statistics_kmeans_plus_plus_df.to_csv('expectation_maximization_statistics_kmeans_plus_plus_df.csv')
expectation_maximization_statistics_kmeans_plus_plus_pca_df.to_csv('expectation_maximization_statistics_kmeans_plus_plus_pca_df.csv')
expectation_maximization_statistics_same_centroid_df.to_csv('expectation_maximization_statistics_same_centroid_df.csv')
error_values_kmeans_same_centroid_df.to_csv('error_values_kmeans_same_centroid_df.csv')


# # Reading the Dataframes

# In[393]:


expectation_maximization_statistics_df=pd.read_csv('expectation_maximization_statistics_df.csv')
expectation_maximization_statistics_kmeans_plus_plus_df=pd.read_csv('expectation_maximization_statistics_kmeans_plus_plus_df.csv')
expectation_maximization_statistics_kmeans_plus_plus_pca_df=pd.read_csv('expectation_maximization_statistics_kmeans_plus_plus_pca_df.csv')
expectation_maximization_statistics_same_centroid_df=pd.read_csv('expectation_maximization_statistics_same_centroid_df.csv')
error_values_kmeans_same_centroid_df=pd.read_csv('error_values_kmeans_same_centroid_df.csv')


# In[394]:


expectation_maximization_statistics_df.head(1)


# In[405]:


import seaborn as sns
expectation_maximization_statistics_df['algorithm']='GMM'
error_values_kmeans_same_centroid_df['algorithm']='K-Means'
comparison_df=pd.DataFrame()
comparison_df=pd.concat([expectation_maximization_statistics_df[['algorithm','No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']],
error_values_kmeans_same_centroid_df[['algorithm','No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']]
],ignore_index=True )
fig, ax = plt.subplots(figsize=(5,5))
sns.boxplot(x='No_of_Clusters', y='within_sum_of_square_error', hue='algorithm',
data=comparison_df[comparison_df['algorithm'].isin (['K-Means','GMM'])],ax=ax);
plt.title('Box Plot of SSE for GMM and K means')
plt.show()


# In[408]:


import seaborn as sns
expectation_maximization_statistics_df['algorithm']='GMM'
error_values_kmeans_same_centroid_df['algorithm']='K-Means'
comparison_df=pd.DataFrame()
comparison_df=pd.concat([expectation_maximization_statistics_df[['algorithm','No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']],
error_values_kmeans_same_centroid_df[['algorithm','No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']]
],ignore_index=True )
fig, ax = plt.subplots(figsize=(5,5))
sns.boxplot(x='No_of_Clusters', y='Calinski_Harbaz_score', hue='algorithm',
data=comparison_df[comparison_df['algorithm'].isin (['K-Means','GMM'])],ax=ax);
plt.title('Box Plot of Calinski Harbaz Score for GMM and K means')
plt.show()


# In[409]:


import seaborn as sns
expectation_maximization_statistics_df['algorithm']='GMM'
error_values_kmeans_same_centroid_df['algorithm']='K-Means'
comparison_df=pd.DataFrame()
comparison_df=pd.concat([expectation_maximization_statistics_df[['algorithm','No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']],
error_values_kmeans_same_centroid_df[['algorithm','No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']]
],ignore_index=True )
fig, ax = plt.subplots(figsize=(5,5))
sns.boxplot(x='No_of_Clusters', y='davies_bouldin_score', hue='algorithm',
data=comparison_df[comparison_df['algorithm'].isin (['K-Means','GMM'])],ax=ax);
plt.title('Box Plot of Davies Bouldin Score for GMM and K means')
plt.show()


# # Other Types of Distributions-Poisson Distribution k++

# In[22]:


df_sample_scaled.columns=df_diabetes.columns


# In[252]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_sample_scaled = pd.DataFrame(scaler.fit_transform(df_sample_scaled), columns=df.columns)
df_sample_scaled


# In[374]:


from scipy.stats import multivariate_normal
from scipy.special import gamma
import numpy as np
import math 

def initialization_of_Poisson(input_dataframe,no_of_clusters):
    input_dataframe_values = input_dataframe.values
    row, column = input_dataframe_values.shape
    means_vector = input_dataframe_values[np.random.choice(input_dataframe_values.shape[0], no_of_clusters,replace=False), :]
    weights_vector = np.ones(no_of_clusters)/no_of_clusters
    return means_vector,weights_vector

def get_poisson(input_dataframe_values,means,weights,no_of_clusters):
    posterior_clusters=np.zeros(no_of_clusters)
    gamma_array=np.array([math.gamma(index+1) for index in input_dataframe_values])
    for cluster in range(no_of_clusters):
        temp=np.exp(-means[cluster])*np.power(means[cluster],input_dataframe_values)/gamma_array
        posterior_clusters[cluster]=weights[cluster]*np.prod(temp)+0.0001
    return posterior_clusters
    

    
def fit_Poisson_mixture_models(input_dataframe,no_of_clusters,max_no_of_iterations,threshold):
    input_dataframe_values = input_dataframe.values
    row, column = input_dataframe_values.shape
    means,weights=initialization_of_Poisson(input_dataframe,no_of_clusters)
    iteration = 0
    previous_log_likelihood_scalar=0
    while iteration < max_no_of_iterations:
        previous_means_df=pd.DataFrame(means)
        posterior_probabilities = np.zeros((len(input_dataframe_values),no_of_clusters))
        for row_number in range(input_dataframe_values.shape[0]):
            posterior_probabilities[row_number]=get_poisson(input_dataframe_values[row_number],means,weights,no_of_clusters)
        posterior_probabilities=np.nan_to_num(posterior_probabilities,nan=0)
        
        for j in range(no_of_clusters):
            weighted_sum = np.zeros((1, means.shape[1]))
            sum_posterior = 0.0
            for i in range(row):
                weighted_sum += posterior_probabilities[i][j] * input_dataframe_values[i]
                sum_posterior += posterior_probabilities[i][j]
            means[j] = weighted_sum/sum_posterior
            weights[j] = np.mean(posterior_probabilities[:, j])
        new_means_df=pd.DataFrame(means)
        euclidean_distance=[]
        for col in new_means_df.columns:
            col_distance = euclidean(previous_means_df[col], new_means_df[col])
            euclidean_distance.append(col_distance)
        threshold_calculated=sum(euclidean_distance)/no_of_clusters
        iteration += 1
        if threshold_calculated<threshold:
            return means,posterior_probabilities
        if iteration>max_no_of_iterations:
            return means,posterior_probabilities
        iteration += 1
        
    return means,posterior_probabilities


# In[375]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(df_diabetes_scaled)
df_diabetes_scaled_min_max=scaler.transform(df_diabetes_scaled)
df_diabetes_scaled_min_max=pd.DataFrame(df_diabetes_scaled_min_max)
expectation_maximization_statistics_poisson=[]
for no_of_clusters in range(2,6):
    print(no_of_clusters)
    for no_of_experiments in range(1,21):
        print(no_of_experiments)
        means,posterior_probabilities=fit_Poisson_mixture_models(df_diabetes_scaled_min_max,no_of_clusters,100,10)
        cluster_labels_original=np.array(pd.DataFrame(posterior_probabilities).idxmax(axis=1))
        cluster_labels_array=np.unique(np.array(pd.DataFrame(posterior_probabilities).idxmax(axis=1)))
        list_of_clusters=np.array([i for i in range(0,no_of_clusters)])
        missing_clusters=set(list_of_clusters)-set(cluster_labels_array)
        for missing_value in missing_clusters:
            unique_values,value_counts=np.unique(cluster_labels_original,return_counts=True)
            values_to_replace=unique_values[value_counts > 1]
            value_to_replace=np.random.choice(values_to_replace)
            indices=np.where(cluster_labels_original==value_to_replace)[0]
            if len(indices)==0:
                indices=[0]
            random_index=np.random.choice(indices)
            new_value=missing_value
            cluster_labels_original[random_index]=new_value
        cluster_labels_array=cluster_labels_original
        within_sum_of_square_error=wcss_emm(df_diabetes_scaled_min_max,cluster_labels_array,no_of_clusters)
        Calinski_Harbaz_score_value=Calinski_Harbaz_score(df_diabetes_scaled_min_max,cluster_labels_array)
        dbs_value=davies_bouldin_score(df_diabetes_scaled_min_max,cluster_labels_array)
        silheoutte_score_value=0
        expectation_maximization_statistics_poisson.append([no_of_clusters,no_of_experiments,within_sum_of_square_error,silheoutte_score_value,Calinski_Harbaz_score_value,dbs_value])
        print("Appended_to_dataframe")
expectation_maximization_statistics_poisson_df= pd.DataFrame(expectation_maximization_statistics_poisson,columns=['No_of_Clusters', 'Iteration Number', 'within_sum_of_square_error','silheoutte_score','Calinski_Harbaz_score','davies_bouldin_score'])


# In[377]:


expectation_maximization_statistics_poisson_df.to_csv('expectation_maximization_statistics_poisson_df.csv')


# # Running code for Exponential Distribution

# In[383]:


from scipy.stats import multivariate_normal
from scipy.special import gamma
import numpy as np
import math 

def initialization_of_exponential(input_dataframe,no_of_clusters):
    input_dataframe_values = input_dataframe.values
    row, column = input_dataframe_values.shape
    means_vector = input_dataframe_values[np.random.choice(input_dataframe_values.shape[0], no_of_clusters,replace=False), :]
    weights_vector = np.ones(no_of_clusters)/no_of_clusters
    return means_vector,weights_vector

def get_exponential(input_dataframe_values,means,weights,no_of_clusters):
    posterior_clusters=np.zeros(no_of_clusters)
    for cluster in range(no_of_clusters):
        mean_temp=1/(means[cluster]+0.01)
        exponential=np.exp(-mean_temp*input_dataframe_values)/mean_temp
        posterior_clusters[cluster] = weights[cluster]*np.prod(exponential)+0.0001
    return posterior_clusters
    

    
def fit_exponential_mixture_models(input_dataframe,no_of_clusters,max_no_of_iterations,threshold):
    input_dataframe_values = input_dataframe.values
    row, column = input_dataframe_values.shape
    means,weights=initialization_of_exponential(input_dataframe,no_of_clusters)
    iteration = 0
    previous_log_likelihood_scalar=0
    while iteration < max_no_of_iterations:
        previous_means_df=pd.DataFrame(means)
        posterior_probabilities = np.zeros((len(input_dataframe_values),no_of_clusters))
        for row_number in range(input_dataframe_values.shape[0]):
            posterior_probabilities[row_number]=get_exponential(input_dataframe_values[row_number],means,weights,no_of_clusters)
        posterior_probabilities=np.nan_to_num(posterior_probabilities,nan=0)
        
        for j in range(no_of_clusters):
            weighted_sum = np.zeros((1, means.shape[1]))
            sum_posterior = 0.0
            for i in range(row):
                weighted_sum += posterior_probabilities[i][j] * input_dataframe_values[i]
                sum_posterior += posterior_probabilities[i][j]
            means[j] = weighted_sum/sum_posterior
            weights[j] = np.mean(posterior_probabilities[:, j])
        new_means_df=pd.DataFrame(means)
        euclidean_distance=[]
        for col in new_means_df.columns:
            col_distance = euclidean(previous_means_df[col], new_means_df[col])
            euclidean_distance.append(col_distance)
        threshold_calculated=sum(euclidean_distance)/no_of_clusters
        iteration += 1
        if threshold_calculated<threshold:
            return means,posterior_probabilities
        if iteration>max_no_of_iterations:
            return means,posterior_probabilities
        iteration += 1
        
    return means,posterior_probabilities


# In[384]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(df_diabetes_scaled)
df_diabetes_scaled_min_max=scaler.transform(df_diabetes_scaled)
df_diabetes_scaled_min_max=pd.DataFrame(df_diabetes_scaled_min_max)
expectation_maximization_statistics_exponential=[]
for no_of_clusters in range(2,6):
    print(no_of_clusters)
    for no_of_experiments in range(1,21):
        print(no_of_experiments)
        means,posterior_probabilities=fit_exponential_mixture_models(df_diabetes_scaled_min_max,no_of_clusters,100,10)
        cluster_labels_original=np.array(pd.DataFrame(posterior_probabilities).idxmax(axis=1))
        cluster_labels_array=np.unique(np.array(pd.DataFrame(posterior_probabilities).idxmax(axis=1)))
        list_of_clusters=np.array([i for i in range(0,no_of_clusters)])
        missing_clusters=set(list_of_clusters)-set(cluster_labels_array)
        for missing_value in missing_clusters:
            unique_values,value_counts=np.unique(cluster_labels_original,return_counts=True)
            values_to_replace=unique_values[value_counts > 1]
            value_to_replace=np.random.choice(values_to_replace)
            indices=np.where(cluster_labels_original==value_to_replace)[0]
            if len(indices)==0:
                indices=[0]
            random_index=np.random.choice(indices)
            new_value=missing_value
            cluster_labels_original[random_index]=new_value
        cluster_labels_array=cluster_labels_original
        within_sum_of_square_error=wcss_emm(df_diabetes_scaled_min_max,cluster_labels_array,no_of_clusters)
        Calinski_Harbaz_score_value=Calinski_Harbaz_score(df_diabetes_scaled_min_max,cluster_labels_array)
        dbs_value=davies_bouldin_score(df_diabetes_scaled_min_max,cluster_labels_array)
        silheoutte_score_value=0
        expectation_maximization_statistics_exponential.append([no_of_clusters,no_of_experiments,within_sum_of_square_error,silheoutte_score_value,Calinski_Harbaz_score_value,dbs_value])
        print("Appended_to_dataframe")
expectation_maximization_statistics_exponential_df= pd.DataFrame(expectation_maximization_statistics_exponential,columns=['No_of_Clusters', 'Iteration Number', 'within_sum_of_square_error','silheoutte_score','Calinski_Harbaz_score','davies_bouldin_score'])


# In[386]:


expectation_maximization_statistics_exponential_df.to_csv('expectation_maximization_statistics_exponential_df.csv')


# # Running GMM on min max scaled dataset

# In[388]:


from scipy.stats import multivariate_normal
import numpy as np
from scipy.spatial.distance import euclidean

def initialization_of_GMM(input_dataframe,no_of_clusters):
    '''
    The function takes scaled dataframe as input and initializes the GMM means,Covariances,and Weights
    '''
    input_dataframe_values = input_dataframe.values
    row, column = input_dataframe_values.shape
    # Randomly initialize means vector
    means_vector = input_dataframe_values[np.random.choice(input_dataframe_values.shape[0], no_of_clusters, replace=False), :]
    # Initialize covariance matrices for each cluster
    covariances_vector = np.array([np.eye(column)] * no_of_clusters)
    # Initialize weights from uniform distribution
    weights_vector = np.ones(no_of_clusters) / no_of_clusters
    return means_vector,covariances_vector,weights_vector
    
    
def fit_Guassian_mixture_models_scaled(input_dataframe,no_of_clusters,max_no_of_iterations,threshold):
    input_dataframe_values = input_dataframe.values
    row, column = input_dataframe_values.shape
    means,covariances,weights=initialization_of_GMM(input_dataframe,no_of_clusters)
    iteration = 0
    previous_log_likelihood_scalar=0
    while iteration < max_no_of_iterations:
       
        new_log_likelihood = 0
        for index in range(no_of_clusters):
            try:
                epsilon_weight=1e-6
                cov_inv = np.linalg.pinv(covariances[index] + np.diag(np.ones(covariances[index].shape[0]) * epsilon_weight))
                new_log_likelihood=new_log_likelihood+weights[index]*multivariate_normal.logpdf(input_dataframe_values,means[index], cov_inv)
            except np.linalg.LinAlgError as e:
                continue
        new_log_likelihood_scalar=np.sum(new_log_likelihood)
        previous_means_df=pd.DataFrame(means)
        
        posterior_probabilities = np.zeros((len(input_dataframe_values),no_of_clusters))
        for index in range(no_of_clusters):
            try:
                cov_inv = np.linalg.pinv(covariances[index],rcond=1e-10)
            except np.linalg.LinAlgError as e:
                continue
            try:
                posterior_probabilities[:,index] = weights[index] * multivariate_normal.pdf(input_dataframe_values, means[index], cov_inv)
            except np.linalg.LinAlgError as e:
                continue
        

        
        
        for j in range(no_of_clusters):
            weighted_sum = np.zeros((1, means.shape[1]))
            sum_posterior = 0.0
            for i in range(row):
                weighted_sum += posterior_probabilities[i][j] * input_dataframe_values[i]
                sum_posterior += posterior_probabilities[i][j]
            means[j] = weighted_sum/sum_posterior
            difference = input_dataframe_values - means[j]
            covariances[j] = np.dot((difference * posterior_probabilities[:, j][:, np.newaxis]).T, difference) / np.sum(posterior_probabilities[:, j])
            covariances[j] += np.diag(np.ones(column) * 1e-6)
            weights[j] = np.mean(posterior_probabilities[:, j])
        
        new_means_df=pd.DataFrame(means)
        
        
        euclidean_distance=[]
        for col in new_means_df.columns:
            col_distance = euclidean(previous_means_df[col], new_means_df[col])
            euclidean_distance.append(col_distance)
        threshold_calculated=sum(euclidean_distance)/no_of_clusters
        
        
        iteration += 1
        if threshold_calculated<threshold:
            return means,posterior_probabilities
        if iteration>max_no_of_iterations:
            return means,posterior_probabilities


# In[389]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(df_diabetes_scaled)
df_diabetes_scaled_min_max=scaler.transform(df_diabetes_scaled)
df_diabetes_scaled_min_max=pd.DataFrame(df_diabetes_scaled_min_max)
expectation_maximization_statistics_gaussian_min_max=[]
for no_of_clusters in range(2,6):
    print(no_of_clusters)
    for no_of_experiments in range(1,21):
        print(no_of_experiments)
        means,posterior_probabilities=fit_Guassian_mixture_models_scaled(df_diabetes_scaled_min_max,no_of_clusters,100,10)
        cluster_labels_original=np.array(pd.DataFrame(posterior_probabilities).idxmax(axis=1))
        cluster_labels_array=np.unique(np.array(pd.DataFrame(posterior_probabilities).idxmax(axis=1)))
        list_of_clusters=np.array([i for i in range(0,no_of_clusters)])
        missing_clusters=set(list_of_clusters)-set(cluster_labels_array)
        for missing_value in missing_clusters:
            unique_values,value_counts=np.unique(cluster_labels_original,return_counts=True)
            values_to_replace=unique_values[value_counts > 1]
            value_to_replace=np.random.choice(values_to_replace)
            indices=np.where(cluster_labels_original==value_to_replace)[0]
            if len(indices)==0:
                indices=[0]
            random_index=np.random.choice(indices)
            new_value=missing_value
            cluster_labels_original[random_index]=new_value
        cluster_labels_array=cluster_labels_original
        within_sum_of_square_error=wcss_emm(df_diabetes_scaled_min_max,cluster_labels_array,no_of_clusters)
        Calinski_Harbaz_score_value=Calinski_Harbaz_score(df_diabetes_scaled_min_max,cluster_labels_array)
        dbs_value=davies_bouldin_score(df_diabetes_scaled_min_max,cluster_labels_array)
        silheoutte_score_value=0
        expectation_maximization_statistics_gaussian_min_max.append([no_of_clusters,no_of_experiments,within_sum_of_square_error,silheoutte_score_value,Calinski_Harbaz_score_value,dbs_value])
        print("Appended_to_dataframe")
expectation_maximization_statistics_gaussian_min_max_df= pd.DataFrame(expectation_maximization_statistics_gaussian_min_max,columns=['No_of_Clusters', 'Iteration Number', 'within_sum_of_square_error','silheoutte_score','Calinski_Harbaz_score','davies_bouldin_score'])


# In[390]:


expectation_maximization_statistics_gaussian_min_max_df.to_csv('expectation_maximization_statistics_gaussian_min_max_df.csv')


# In[391]:


expectation_maximization_statistics_gaussian_min_max_df


# # Plotting the graphs

# In[436]:


import seaborn as sns
expectation_maximization_statistics_gaussian_min_max_df['Mixture']='Gaussian'
expectation_maximization_statistics_exponential_df['Mixture']='Exponential'
expectation_maximization_statistics_poisson_df['Mixture']='Poisson'
comparison_df=pd.DataFrame()
comparison_df=pd.concat([expectation_maximization_statistics_gaussian_min_max_df[['Mixture','No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']],
expectation_maximization_statistics_exponential_df[['Mixture','No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']],
expectation_maximization_statistics_poisson_df[['Mixture','No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']]                         
],ignore_index=True )
fig, ax = plt.subplots(figsize=(8,6))
sns.boxplot(x='No_of_Clusters', y='within_sum_of_square_error', hue='Mixture',
data=comparison_df[comparison_df['Mixture'].isin (['Gaussian','Exponential','Poisson'])],ax=ax);
plt.title('Box Plot of within_sum_of_square_error for Different Mixture Modles')
plt.show()


# In[439]:


import seaborn as sns
expectation_maximization_statistics_gaussian_min_max_df['Mixture']='Gaussian'
expectation_maximization_statistics_exponential_df['Mixture']='Exponential'
expectation_maximization_statistics_poisson_df['Mixture']='Poisson'
comparison_df=pd.DataFrame()
comparison_df=pd.concat([expectation_maximization_statistics_gaussian_min_max_df[['Mixture','No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']],
expectation_maximization_statistics_exponential_df[['Mixture','No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']],
expectation_maximization_statistics_poisson_df[['Mixture','No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']]                         
],ignore_index=True )
fig, ax = plt.subplots(figsize=(8,6))
sns.boxplot(x='No_of_Clusters', y='Calinski_Harbaz_score', hue='Mixture',
data=comparison_df[comparison_df['Mixture'].isin (['Gaussian','Exponential','Poisson'])],ax=ax);
plt.title('Box Plot of Calinski_Harbaz_score for Different Mixture Modles')
plt.show()


# In[ ]:


import seaborn as sns
expectation_maximization_statistics_gaussian_min_max_df['Mixture']='Gaussian'
expectation_maximization_statistics_exponential_df['Mixture']='Exponential'
expectation_maximization_statistics_poisson_df['Mixture']='Poisson'
comparison_df=pd.DataFrame()
comparison_df=pd.concat([expectation_maximization_statistics_gaussian_min_max_df[['Mixture','No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']],
expectation_maximization_statistics_exponential_df[['Mixture','No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']],
expectation_maximization_statistics_poisson_df[['Mixture','No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']]                         
],ignore_index=True )
fig, ax = plt.subplots(figsize=(8,6))
sns.boxplot(x='No_of_Clusters', y='davies_bouldin_score', hue='Mixture',
data=comparison_df[comparison_df['Mixture'].isin (['Gaussian','Exponential','Poisson'])],ax=ax);
plt.title('Box Plot of Davies Bouldin Score for Different Mixture Modles')
plt.show()


# # Running EM algorithm without updating covariances and means

# In[159]:


from scipy.stats import multivariate_normal
import numpy as np
from scipy.spatial.distance import euclidean

def initialization_of_GMM(input_dataframe,no_of_clusters):
    '''
    The function takes scaled dataframe as input and initializes the GMM means,Covariances,and Weights
    '''
    input_dataframe_values = input_dataframe.values
    row, column = input_dataframe_values.shape
    # Randomly initialize means vector
    means_vector = input_dataframe_values[np.random.choice(input_dataframe_values.shape[0], no_of_clusters, replace=False), :]
    # Initialize covariance matrices for each cluster
    covariances_vector = np.array([np.eye(column)] * no_of_clusters)
    # Initialize weights from uniform distribution
    weights_vector = np.ones(no_of_clusters) / no_of_clusters
    return means_vector,covariances_vector,weights_vector
    
    
def fit_Guassian_mixture_models_without_covariances(input_dataframe,no_of_clusters,max_no_of_iterations,threshold):
    input_dataframe_values = input_dataframe.values
    row, column = input_dataframe_values.shape
    means,covariances,weights=initialization_of_GMM(input_dataframe,no_of_clusters)
    iteration = 0
    previous_log_likelihood_scalar=0
    while iteration < max_no_of_iterations:
       
        new_log_likelihood = 0
        for index in range(no_of_clusters):
            try:
                epsilon_weight=1e-6
                cov_inv = np.linalg.pinv(covariances[index] + np.diag(np.ones(covariances[index].shape[0]) * epsilon_weight))
                new_log_likelihood=new_log_likelihood+weights[index]*multivariate_normal.logpdf(input_dataframe_values,means[index], cov_inv)
            except np.linalg.LinAlgError as e:
                continue
        new_log_likelihood_scalar=np.sum(new_log_likelihood)
        previous_means_df=pd.DataFrame(means)
        
        posterior_probabilities = np.zeros((len(input_dataframe_values),no_of_clusters))
        for index in range(no_of_clusters):
            try:
                cov_inv = np.linalg.pinv(covariances[index],rcond=1e-10)
            except np.linalg.LinAlgError as e:
                continue
            try:
                posterior_probabilities[:,index] = weights[index] * multivariate_normal.pdf(input_dataframe_values, means[index], cov_inv)
            except np.linalg.LinAlgError as e:
                continue
        

        
        
        for j in range(no_of_clusters):
            weighted_sum = np.zeros((1, means.shape[1]))
            sum_posterior = 0.0
            for i in range(row):
                weighted_sum += posterior_probabilities[i][j] * input_dataframe_values[i]
                sum_posterior += posterior_probabilities[i][j]
            means[j] = weighted_sum/sum_posterior
            
        
        new_means_df=pd.DataFrame(means)
        
        
        euclidean_distance=[]
        for col in new_means_df.columns:
            col_distance = euclidean(previous_means_df[col], new_means_df[col])
            euclidean_distance.append(col_distance)
        threshold_calculated=sum(euclidean_distance)/no_of_clusters
        
        
        iteration += 1
        if threshold_calculated<threshold:
            return means,posterior_probabilities
        if iteration>max_no_of_iterations:
            return means,posterior_probabilities


# In[161]:


expectation_maximization_statistics_without_updation=[]
for no_of_clusters in range(2,6):
    print(no_of_clusters)
    for no_of_experiments in range(1,21):
        print(no_of_experiments)
        means,posterior_probabilities=fit_Guassian_mixture_models_without_covariances(df_diabetes_scaled,no_of_clusters,100,10)
        cluster_labels_original=np.array(pd.DataFrame(posterior_probabilities).idxmax(axis=1))
        cluster_labels_array=np.unique(np.array(pd.DataFrame(posterior_probabilities).idxmax(axis=1)))
        list_of_clusters=np.array([i for i in range(0,no_of_clusters)])
        missing_clusters=set(list_of_clusters)-set(cluster_labels_array)
        for missing_value in missing_clusters:
            unique_values,value_counts=np.unique(cluster_labels_original,return_counts=True)
            values_to_replace=unique_values[value_counts > 1]
            value_to_replace=np.random.choice(values_to_replace)
            indices=np.where(cluster_labels_original==value_to_replace)[0]
            if len(indices)==0:
                indices=[0]
            random_index=np.random.choice(indices)
            new_value=missing_value
            cluster_labels_original[random_index]=new_value
        cluster_labels_array=cluster_labels_original
        within_sum_of_square_error=wcss_emm(df_diabetes_scaled,cluster_labels_array,no_of_clusters)
        Calinski_Harbaz_score_value=Calinski_Harbaz_score(df_diabetes_scaled,cluster_labels_array)
        dbs_value=davies_bouldin_score(df_diabetes_scaled,cluster_labels_array)
        silheoutte_score_value=0
        expectation_maximization_statistics_without_updation.append([no_of_clusters,no_of_experiments,within_sum_of_square_error,silheoutte_score_value,Calinski_Harbaz_score_value,dbs_value])
        print("Appended_to_dataframe")
expectation_maximization_statistics_without_updation_df= pd.DataFrame(expectation_maximization_statistics_without_updation,columns=['No_of_Clusters', 'Iteration Number', 'within_sum_of_square_error','silheoutte_score','Calinski_Harbaz_score','davies_bouldin_score'])


# In[164]:


expectation_maximization_statistics_without_updation_df.to_csv('expectation_maximization_statistics_without_updation_df.csv')


# In[411]:


import seaborn as sns
expectation_maximization_statistics_without_updation_df['algorithm']='GMM'
error_values_kmeans_same_centroid_df['algorithm']='K-Means'
comparison_df=pd.DataFrame()
comparison_df=pd.concat([expectation_maximization_statistics_without_updation_df[['algorithm','No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']],
error_values_kmeans_same_centroid_df[['algorithm','No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']]
],ignore_index=True )
fig, ax = plt.subplots(figsize=(5,5))
sns.boxplot(x='No_of_Clusters', y='within_sum_of_square_error', hue='algorithm',
data=comparison_df[comparison_df['algorithm'].isin (['K-Means','GMM'])],ax=ax);
plt.title('Box Plot of Davies Bouldin Score for GMM and K means')
plt.show()


# In[410]:


import seaborn as sns
expectation_maximization_statistics_without_updation_df['algorithm']='GMM'
error_values_kmeans_same_centroid_df['algorithm']='K-Means'
comparison_df=pd.DataFrame()
comparison_df=pd.concat([expectation_maximization_statistics_without_updation_df[['algorithm','No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']],
error_values_kmeans_same_centroid_df[['algorithm','No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']]
],ignore_index=True )
fig, ax = plt.subplots(figsize=(5,5))
sns.boxplot(x='No_of_Clusters', y='davies_bouldin_score', hue='algorithm',
data=comparison_df[comparison_df['algorithm'].isin (['K-Means','GMM'])],ax=ax);
plt.title('Box Plot of Davies Bouldin Score for GMM and K means')
plt.show()


# In[412]:


import seaborn as sns
expectation_maximization_statistics_without_updation_df['algorithm']='GMM'
error_values_kmeans_same_centroid_df['algorithm']='K-Means'
comparison_df=pd.DataFrame()
comparison_df=pd.concat([expectation_maximization_statistics_without_updation_df[['algorithm','No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']],
error_values_kmeans_same_centroid_df[['algorithm','No_of_Clusters','within_sum_of_square_error','Calinski_Harbaz_score','davies_bouldin_score']]
],ignore_index=True )
fig, ax = plt.subplots(figsize=(5,5))
sns.boxplot(x='No_of_Clusters', y='Calinski_Harbaz_score', hue='algorithm',
data=comparison_df[comparison_df['algorithm'].isin (['K-Means','GMM'])],ax=ax);
plt.title('Box Plot of Davies Bouldin Score for GMM and K means')
plt.show()


# In[ ]:




