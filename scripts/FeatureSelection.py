import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
#%%" open resulting dataset file
traffic_df = pd.read_csv('D:/pythonProject/CICIDS2017-datast-analysis/data/Whole-Traffic.pcap_ISCX.csv')
traffic_df.drop(columns=traffic_df.columns[0], axis=1, inplace=True)
# -----------------------------------------------------------------------------------------------------------------------
# Data cleaning
#%%" removing columns with only one percent of unique values
counts = traffic_df.loc[:, : 'Idle Min'].nunique()
to_del = [i for i, v in enumerate(counts) if (float(v)/traffic_df.loc[:, : 'Idle Min'].shape[0]*100) < 1]
print('Columns with only one percent of unique values: {}'.format(traffic_df.columns[to_del]))
print('Shape before deleting columns: {}'.format(traffic_df.loc[:, : 'Idle Min'].shape))
traffic_df.drop(traffic_df.columns[to_del], axis=1, inplace=True)
print('Shape after deleting columns: {}'.format(traffic_df.loc[:, : 'Idle Min'].shape))
#%%" removing rows with infinite values
print('Number of infinite values: {}'.format(np.isinf(traffic_df.loc[:, : 'Idle Min']).values.sum()))
print('Indexes of infinite values: {}'.format(np.where(np.isinf(traffic_df.loc[:, : 'Idle Min']))))
print('Shape before infinite values deleting: {}'.format(traffic_df.loc[:, : 'Idle Min'].shape))
traffic_df.replace([np.inf, -np.inf], np.nan, inplace=True)
traffic_df.dropna(axis=0, inplace=True)
traffic_df.reset_index(drop=True, inplace=True)
print('Shape after infinite values deleting: {}'.format(traffic_df.loc[:, : 'Idle Min'].shape))
# -----------------------------------------------------------------------------------------------------------------------
# ANOVA f-test feature selection approach
#%%" checking normality assumption
# the data for each factor level (each class in the output feature) should be normally distributed


def plot_data_distribution(dataframe, title, graph_colors):
    font = {'family': 'Times New Roman', 'weight': 'book', 'size': 15}
    plt.rc('font', **font)

    fig, ax = plt.subplots(7, 7, sharey='row', sharex='col', figsize=(15, 15))
    ax = ax.flatten()

    for idx, (columnName, columnData) in enumerate(dataframe.iteritems()):
        n, bins, patches = ax[idx].hist(columnData.values, color=graph_colors[0],
                                        edgecolor=graph_colors[1], alpha=0.8)
        x = bins[:-1] + (bins[1] - bins[0])/2
        ax[idx].plot(x, n, '--', color=graph_colors[2])
        ax[idx].set_yscale('symlog')

    fig.text(0.5, 0.9, title, ha='center', fontsize=35)
    fig.text(0.5, 0.06, 'Value', ha='center', fontsize=30)
    fig.text(0.06, 0.5, 'Frequency', va='center', rotation='vertical', fontsize=30)

    plt.show()


# plot feature distribution after transformation
plot_data_distribution(traffic_df.loc[:, : 'Idle Min'], 'Dataset features distribution before transformation',
                       graph_colors=['salmon', 'red', 'maroon'])

# map data distribution to normal or approximately normal
scaler = QuantileTransformer(output_distribution='normal', n_quantiles=1000, random_state=42)
scaled_df = scaler.fit_transform(traffic_df.loc[:, : 'Idle Min'])
x_anova_df = pd.DataFrame(scaled_df)
# plot feature distribution after transformation
plot_data_distribution(x_anova_df, 'Dataset features distribution after transformation',
                       graph_colors=['mediumaquamarine', 'teal', 'blue'])
#%%" performing ANOVA f-test
y_df = traffic_df.iloc[:, -1]
# assign column (features) name to the x_df
x_anova_df.columns = traffic_df.loc[:, : 'Idle Min'].columns

fv, pv = f_classif(x_anova_df, y_df)
# print features f-scores
for i in range(len(fv)):
    print('Feature f-score {}: {}'.format(x_anova_df.columns[i], fv[i]))


def plot_scores(x, y, title, line):
    font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 15}
    plt.rc('font', **font)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x, y, color='lightpink')
    ax.axhline(line, ls='--', color='r')
    ax.get_xaxis().set_visible(False)
    ax.set(ylabel='Score')
    ax.set_title(title, fontsize=20)

    plt.show()


plot_scores(x_anova_df.columns, fv, "Distribution of features' f-scores", 1500)

#%%"select features with the score equal or more then 1500


# function drops features with the scorers less than threshold and returns dict with selected features and scores
def select_features_by_threshold(df, features_values, threshold):
    features_dict = dict()
    for i in range(len(features_values)):
        features_dict[df.columns[i]] = features_values[i]

    to_del = [key for key, v in features_dict.items() if v <= threshold]
    print('Features to delete count: {}'.format(len(to_del)))
    print('Features to delete: {}'.format(to_del))
    print('Shape before selecting features: {}'.format(df.shape))
    df.drop(to_del, axis=1, inplace=True)
    print('Shape after selecting features: {}'.format(df.shape))

    selected_features_dict = features_dict
    for i in to_del:
        del selected_features_dict[i]
    print('Selected features: {}'.format(selected_features_dict))
    print('Selected features number: {}'.format(len(selected_features_dict)))

    return selected_features_dict


selected_features_dict = select_features_by_threshold(x_anova_df, fv, 1500)


#%%"plot selected features


def plot_selected_features(x, y, title, color):
    font = {'family': 'Times New Roman', 'weight': 'book', 'size': 20}
    plt.rc('font', **font)
    plt.rcParams.update({'figure.autolayout': True})

    fig, ax = plt.subplots(figsize=(12, 15))

    ax.barh(x, y, color=color)
    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=45, horizontalalignment='right')
    ax.set_xlabel('Score', fontsize=25)
    ax.set_ylabel('Feature', fontsize=25)
    ax.set_title(title, fontsize=30)

    plt.show()


plot_selected_features(list(selected_features_dict.keys()), list(selected_features_dict.values()),
                       "Distribution of selected features' f-scores", 'crimson')

# -----------------------------------------------------------------------------------------------------------------------
# Information gain feature selection approach
#%%" applying information gain approach
x_info_gain_df = traffic_df.loc[:, : 'Idle Min']
mi = mutual_info_classif(x_info_gain_df, y_df)

# print features info gain scores
for i in range(len(mi)):
    print('Feature info gain score {}: {}'.format(x_info_gain_df.columns[i], mi[i]))

plot_scores(x_info_gain_df.columns, mi, "Distribution of features' information gain scores", 0.5)
selected_features_dict = select_features_by_threshold(x_info_gain_df, mi, 0.5)
plot_selected_features(list(selected_features_dict.keys()), list(selected_features_dict.values()),
                       "Distribution of selected features' info gain scores", 'lightpink')
# -----------------------------------------------------------------------------------------------------------------------
# removing highly correlated with each other input features
#%%" calculating Pearson's correlation


# function plots heatmap and returns Pearson's correlation matrix
def plot_heatmap(df, cmap):
    cor_matrix = df.corr()
    mask = np.triu(np.ones_like(cor_matrix, dtype=np.bool))
    plt.figure(figsize=(40, 25))
    heatmap = sns.heatmap(cor_matrix, mask=mask, vmin=-1, vmax=1, annot=True, cmap=cmap)
    heatmap.set_title('Triangle Correlation Heatmap', fontdict={'fontsize': 50}, pad=12)
    plt.show()

    return cor_matrix


# correlation matrix and heatmap for ANOVA f-test approach
cor_matrix_anova = plot_heatmap(x_anova_df, 'BrBG')

# correlation matrix and heatmap for info gain approach
cor_matrix_info_gain = plot_heatmap(x_info_gain_df, 'BuPu')

#%%" delete highly correlated columns from the dataset


def drop_cor_columns(df, cor_matrix):
    upper_tri = cor_matrix.abs().where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    to_del = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
    print('Number of highly correlated features: {}'.format(len(to_del)))
    print('Highly correlated features: {}'.format(to_del))
    print('Shape before highly correlated features deleting: {}'.format(df.shape))
    df.drop(to_del, axis=1, inplace=True)
    print('Shape after highly correlated features deleting: {}'.format(df.shape))


# drop columns in x_anova_df (ANOVA f-test approach)
drop_cor_columns(x_anova_df, cor_matrix_anova)

# drop columns in x_info_gain_df (Information gain approach)
drop_cor_columns(x_info_gain_df, cor_matrix_info_gain)

#%%" save datasets

info_gain_df = pd.concat([x_info_gain_df, y_df], axis=1)
info_gain_df.to_csv(r'D:/pythonProject/CICIDS2017-datast-analysis/data/Processed-Traffic(info gain).pcap_ISCX.csv',
                    index=False)

anova_df = pd.concat([x_anova_df, y_df], axis=1)
anova_df.to_csv(r'D:/pythonProject/CICIDS2017-datast-analysis/data/Processed-Traffic(anova).pcap_ISCX.csv',
                index=False)

traffic_df.to_csv(r'D:/pythonProject/CICIDS2017-datast-analysis/data/Processed-Traffic(only cleaned).pcap_ISCX.csv',
                  index=False)






