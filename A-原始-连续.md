

```python
# 合并训练集合与测试集合
# 1. 读取数据
headers = [
    'age', 
    'workclass', 
    'fnlwgt', 
    'education', 
    'education-num', 
    'marital-status', 
    'occupation', 
    'relation', 
    'race', 
    'sex', 
    'capital-gain', 
    'capital-loss', 
    'hours-per-week', 
    'native-country',
    'predcalss']

import time
import pandas as pd
# 2.合并数据
train_raw = pd.read_csv(
    'adult.data',
    header=None,
    names=headers,
    sep=',\s',
    na_values='?',
    engine='python',
    skiprows=1
)
test_raw = pd.read_csv(
    'adult.test',
    header=None,
    names=headers,
    sep=',\s',
    engine='python',
    na_values='?',
    skiprows=1
)
print('数据加载完毕')
dataset_raw = train_raw.append(test_raw)
# 3. 合并的数据集重新索引
# 重新索引
dataset_raw.reset_index(inplace=True)
# 删除索引
dataset_raw.drop('index', axis=1,inplace=True)
print('merge over')
data_bin = pd.DataFrame()  # 处理特征后的数据集（离散数据集）
data_con = pd.DataFrame() # 连续数据集

# 处理标签特征
#    - >50k, 1
#    - <=50,  0
dataset_raw.loc[dataset_raw['predcalss'] == '>50K', 'predcalss' ] = 1
dataset_raw.loc[dataset_raw['predcalss'] == '<=50K', 'predcalss'] = 0

dataset_raw.loc[dataset_raw['predcalss'] == '>50K.', 'predcalss'] = 1
dataset_raw.loc[dataset_raw['predcalss'] == '<=50K.', 'predcalss' ] = 0

# 处理好的特征拷贝到处理好后缓冲
data_bin['predcalss'] = dataset_raw['predcalss']
data_con['predcalss'] = dataset_raw['predcalss']
print('feature label over')
# age的特征处理，1.连续性数据，不处理，2.离散化处理：年龄分成不同的年龄段
data_con['age'] = dataset_raw['age']   # conitnue
data_bin['age'] = pd.cut(dataset_raw['age'], 10)   # bin
print('feature age over')
# 工作岗位的分类重新合并
dataset_raw.loc[dataset_raw['workclass']=='Without-pay','workclass'] = 'Not Working'
dataset_raw.loc[dataset_raw['workclass']=='Never-worked','workclass'] = 'Not Working'
dataset_raw.loc[dataset_raw['workclass']=='Federal-gov','workclass']  = 'Fed-gov'
dataset_raw.loc[dataset_raw['workclass']=='State-gov','workclass']  = 'Non-Fed-gov'
dataset_raw.loc[dataset_raw['workclass']=='Local-gov','workclass']  = 'Non-Fed-gov'
dataset_raw.loc[dataset_raw['workclass']=='Self-emp-not-inc','workclass'] = 'Self-emp'
dataset_raw.loc[dataset_raw['workclass']=='Self-emp-inc','workclass']  = 'Self-emp'

# 拷贝到训练缓冲
data_bin['workclass'] = dataset_raw['workclass']
data_con['workclass'] = dataset_raw['workclass']
print('feature workclass over')

# 对职业进行合并
dataset_raw.loc[dataset_raw['occupation']=='Adm-clerical', 'occupation'] = 'Admin'
dataset_raw.loc[dataset_raw['occupation']=='Armed-Forces', 'occupation'] = 'Military'
dataset_raw.loc[dataset_raw['occupation']=='Craft-repair', 'occupation'] = 'Manual Labour'
dataset_raw.loc[dataset_raw['occupation']=='Exec-managerial', 'occupation'] = 'Office Labour'
dataset_raw.loc[dataset_raw['occupation']=='Farming-fishing', 'occupation'] = 'Manual Labour'
dataset_raw.loc[dataset_raw['occupation']=='Handlers-cleaners', 'occupation'] = 'Manual Labour'
dataset_raw.loc[dataset_raw['occupation']=='Machine-op-inspct', 'occupation'] = 'Manual Labour'
dataset_raw.loc[dataset_raw['occupation']=='Other-service', 'occupation'] = 'Service'
dataset_raw.loc[dataset_raw['occupation']=='Priv-house-serv', 'occupation'] = 'Service'
dataset_raw.loc[dataset_raw['occupation']=='Prof-specialty', 'occupation'] = 'Professional'
dataset_raw.loc[dataset_raw['occupation']=='Protective-serv', 'occupation'] = 'Military'
dataset_raw.loc[dataset_raw['occupation']=='Sales', 'occupation'] = 'Office Labour'
dataset_raw.loc[dataset_raw['occupation']=='Tech-support', 'occupation'] = 'Office Labour'
dataset_raw.loc[dataset_raw['occupation']=='Transport-moving', 'occupation'] = 'Manual Labour'

data_bin['occupation'] = dataset_raw['occupation']
data_con['occupation'] = dataset_raw['occupation']
print('feature occupation over')
dataset_raw.loc[dataset_raw['native-country'] == 'Cambodia'                    , 'native-country'] = 'SE-Asia'
dataset_raw.loc[dataset_raw['native-country'] == 'Canada'                      , 'native-country'] = 'British-Commonwealth'    
dataset_raw.loc[dataset_raw['native-country'] == 'China'                       , 'native-country'] = 'China'       
dataset_raw.loc[dataset_raw['native-country'] == 'Columbia'                    , 'native-country'] = 'South-America'    
dataset_raw.loc[dataset_raw['native-country'] == 'Cuba'                        , 'native-country'] = 'South-America'        
dataset_raw.loc[dataset_raw['native-country'] == 'Dominican-Republic'          , 'native-country'] = 'South-America'
dataset_raw.loc[dataset_raw['native-country'] == 'Ecuador'                     , 'native-country'] = 'South-America'     
dataset_raw.loc[dataset_raw['native-country'] == 'El-Salvador'                 , 'native-country'] = 'South-America' 
dataset_raw.loc[dataset_raw['native-country'] == 'England'                     , 'native-country'] = 'British-Commonwealth'
dataset_raw.loc[dataset_raw['native-country'] == 'France'                      , 'native-country'] = 'Euro_Group_1'
dataset_raw.loc[dataset_raw['native-country'] == 'Germany'                     , 'native-country'] = 'Euro_Group_1'
dataset_raw.loc[dataset_raw['native-country'] == 'Greece'                      , 'native-country'] = 'Euro_Group_2'
dataset_raw.loc[dataset_raw['native-country'] == 'Guatemala'                   , 'native-country'] = 'South-America'
dataset_raw.loc[dataset_raw['native-country'] == 'Haiti'                       , 'native-country'] = 'South-America'
dataset_raw.loc[dataset_raw['native-country'] == 'Holand-Netherlands'          , 'native-country'] = 'Euro_Group_1'
dataset_raw.loc[dataset_raw['native-country'] == 'Honduras'                    , 'native-country'] = 'South-America'
dataset_raw.loc[dataset_raw['native-country'] == 'Hong'                        , 'native-country'] = 'China'
dataset_raw.loc[dataset_raw['native-country'] == 'Hungary'                     , 'native-country'] = 'Euro_Group_2'
dataset_raw.loc[dataset_raw['native-country'] == 'India'                       , 'native-country'] = 'British-Commonwealth'
dataset_raw.loc[dataset_raw['native-country'] == 'Iran'                        , 'native-country'] = 'Euro_Group_2'
dataset_raw.loc[dataset_raw['native-country'] == 'Ireland'                     , 'native-country'] = 'British-Commonwealth'
dataset_raw.loc[dataset_raw['native-country'] == 'Italy'                       , 'native-country'] = 'Euro_Group_1'
dataset_raw.loc[dataset_raw['native-country'] == 'Jamaica'                     , 'native-country'] = 'South-America'
dataset_raw.loc[dataset_raw['native-country'] == 'Japan'                       , 'native-country'] = 'APAC'
dataset_raw.loc[dataset_raw['native-country'] == 'Laos'                        , 'native-country'] = 'SE-Asia'
dataset_raw.loc[dataset_raw['native-country'] == 'Mexico'                      , 'native-country'] = 'South-America'
dataset_raw.loc[dataset_raw['native-country'] == 'Nicaragua'                   , 'native-country'] = 'South-America'
dataset_raw.loc[dataset_raw['native-country'] == 'Outlying-US(Guam-USVI-etc)'  , 'native-country'] = 'South-America'
dataset_raw.loc[dataset_raw['native-country'] == 'Peru'                        , 'native-country'] = 'South-America'
dataset_raw.loc[dataset_raw['native-country'] == 'Philippines'                 , 'native-country'] = 'SE-Asia'
dataset_raw.loc[dataset_raw['native-country'] == 'Poland'                      , 'native-country'] = 'Euro_Group_2'
dataset_raw.loc[dataset_raw['native-country'] == 'Portugal'                    , 'native-country'] = 'Euro_Group_2'
dataset_raw.loc[dataset_raw['native-country'] == 'Puerto-Rico'                 , 'native-country'] = 'South-America'
dataset_raw.loc[dataset_raw['native-country'] == 'Scotland'                    , 'native-country'] = 'British-Commonwealth'
dataset_raw.loc[dataset_raw['native-country'] == 'South'                       , 'native-country'] = 'Euro_Group_2'
dataset_raw.loc[dataset_raw['native-country'] == 'Taiwan'                      , 'native-country'] = 'China'
dataset_raw.loc[dataset_raw['native-country'] == 'Thailand'                    , 'native-country'] = 'SE-Asia'
dataset_raw.loc[dataset_raw['native-country'] == 'Trinadad&Tobago'             , 'native-country'] = 'South-America'
dataset_raw.loc[dataset_raw['native-country'] == 'United-States'               , 'native-country'] = 'United-States'
dataset_raw.loc[dataset_raw['native-country'] == 'Vietnam'                     , 'native-country'] = 'SE-Asia'
dataset_raw.loc[dataset_raw['native-country'] == 'Yugoslavia'                  , 'native-country'] = 'Euro_Group_2'

data_bin['native-country'] = dataset_raw['native-country']
data_con['native-country'] = dataset_raw['native-country']
print('feature counyrt over')
dataset_raw.loc[dataset_raw['education'] == '10th'          , 'education'] = 'Dropout'
dataset_raw.loc[dataset_raw['education'] == '11th'          , 'education'] = 'Dropout'
dataset_raw.loc[dataset_raw['education'] == '12th'          , 'education'] = 'Dropout'
dataset_raw.loc[dataset_raw['education'] == '1st-4th'       , 'education'] = 'Dropout'
dataset_raw.loc[dataset_raw['education'] == '5th-6th'       , 'education'] = 'Dropout'
dataset_raw.loc[dataset_raw['education'] == '7th-8th'       , 'education'] = 'Dropout'
dataset_raw.loc[dataset_raw['education'] == '9th'           , 'education'] = 'Dropout'
dataset_raw.loc[dataset_raw['education'] == 'Assoc-acdm'    , 'education'] = 'Associate'
dataset_raw.loc[dataset_raw['education'] == 'Assoc-voc'     , 'education'] = 'Associate'
dataset_raw.loc[dataset_raw['education'] == 'Bachelors'     , 'education'] = 'Bachelors'
dataset_raw.loc[dataset_raw['education'] == 'Doctorate'     , 'education'] = 'Doctorate'
dataset_raw.loc[dataset_raw['education'] == 'HS-Grad'       , 'education'] = 'HS-Graduate'
dataset_raw.loc[dataset_raw['education'] == 'Masters'       , 'education'] = 'Masters'
dataset_raw.loc[dataset_raw['education'] == 'Preschool'     , 'education'] = 'Dropout'
dataset_raw.loc[dataset_raw['education'] == 'Prof-school'   , 'education'] = 'Professor'
dataset_raw.loc[dataset_raw['education'] == 'Some-college'  , 'education'] = 'HS-Graduate'

data_bin['education'] = dataset_raw['education']
data_con['education'] = dataset_raw['education']
print('feature edu over')
dataset_raw.loc[dataset_raw['marital-status'] == 'Never-married'        , 'marital-status'] = 'Never-Married'
dataset_raw.loc[dataset_raw['marital-status'] == 'Married-AF-spouse'    , 'marital-status'] = 'Married'
dataset_raw.loc[dataset_raw['marital-status'] == 'Married-civ-spouse'   , 'marital-status'] = 'Married'
dataset_raw.loc[dataset_raw['marital-status'] == 'Married-spouse-absent', 'marital-status'] = 'Not-Married'
dataset_raw.loc[dataset_raw['marital-status'] == 'Separated'            , 'marital-status'] = 'Separated'
dataset_raw.loc[dataset_raw['marital-status'] == 'Divorced'             , 'marital-status'] = 'Separated'
dataset_raw.loc[dataset_raw['marital-status'] == 'Widowed'              , 'marital-status'] = 'Widowed'

data_bin['marital-status'] = dataset_raw['marital-status']
data_con['marital-status'] = dataset_raw['marital-status']

print('feature status over')
# fnlw:final weight:序号
# 离散：连续
data_bin['fnlwgt'] = pd.cut(dataset_raw['fnlwgt'], 10)
data_con['fnlwgt'] = dataset_raw['fnlwgt']
print('feature fntwgt over')
# education-num ：连续，离散

data_bin['education-num'] = pd.cut(dataset_raw['education-num'], 10)
data_con['education-num'] = dataset_raw['education-num']
print('feature edu-num over')

# hours-per-week
data_bin['hours-per-week'] = pd.cut(dataset_raw['hours-per-week'], 10)
data_con['hours-per-week'] = dataset_raw['hours-per-week']
print('feature hours over')
# capital-gain
data_bin['capital-gain'] = pd.cut(dataset_raw['capital-gain'], 10)
data_con['capital-gain'] = dataset_raw['capital-gain']
print('feature gain over')

data_bin['capital-loss'] = pd.cut(dataset_raw['capital-loss'], 10)
data_con['capital-loss'] = dataset_raw['capital-loss']
print('feature losss over')

# race ,sex, relationship
data_bin['race'] = data_con['race'] = dataset_raw['race']
data_bin['sex'] = data_con['sex'] = dataset_raw['sex']
data_bin['relation'] = data_con['relation'] = dataset_raw['relation']
print('feature race sex rel over')
# age + hour-per-week
data_con['age-hours'] = data_con['age'] * data_con['hours-per-week']

data_bin['age-hours'] = pd.cut(data_con['age-hours'], 10)

# sex + married
data_bin['sex-marital'] = data_con['sex-marital'] = data_con['sex'] + data_con['marital-status']
print('feature creation  over')
# 确定编码的字段
one_hot_cols = data_bin.columns.tolist()
# 标签的字段不需要编码
one_hot_cols.remove('predcalss')

data_bin_encode = pd.get_dummies(data_bin, columns=one_hot_cols)

# 连续的特征处理
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
# 转换数据类型
data_con = data_con.astype('str')
# 编码转换
data_con_encode = data_con.apply(encoder.fit_transform)

print('encode over')
# 数据差分成训练数据，测试数据集。

train = data_con_encode.loc[0:32560,:]
test = data_con_encode.loc[32560:, :]

train = train.dropna(axis=0)
test= test.dropna(axis=0)
print('dataset split over')
# --------------------------------------------------------------
from sklearn.svm import SVC

# 准备数据集
X_train = train.drop('predcalss',axis=1)
Y_train =train['predcalss'].astype('int64')  # 转换为数值型

X_test = test.drop('predcalss',axis=1)
Y_test =test['predcalss'].astype('int64')  # 转换为数值型


# ---------------------------------------------
# 逻辑回归
start = time.time()
from sklearn.linear_model import LogisticRegression

lr_classifier = LogisticRegression(solver='lbfgs',C=1.0, max_iter=10000)

lr_classifier.fit(X_train,Y_train)
train_score = lr_classifier.score(X_train,Y_train)
test_score = lr_classifier.score(X_test,Y_test)
end = time.time()
print(F"训练集准确率：{train_score*100:8.2f}%，测试集准确率：{test_score*100:8.2f}%，Time:{end-start}s")

# -----------------------------------------
# k近邻算法
start = time.time()
from sklearn.neighbors import KNeighborsClassifier

knn_classifier = KNeighborsClassifier(n_neighbors=3)

knn_classifier.fit(X_train,Y_train)
train_score = knn_classifier.score(X_train,Y_train)
test_score = knn_classifier.score(X_test,Y_test)
end = time.time()
print(F"训练集准确率：{train_score*100:8.2f}%，测试集准确率：{test_score*100:8.2f}%，Time:{end-start}s")


# ---------------------------------------------

# 朴素贝叶斯分类
start = time.time()
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
gau_classifier = GaussianNB()

gau_classifier.fit(X_train,Y_train)
train_score = gau_classifier.score(X_train,Y_train)
test_score = gau_classifier.score(X_test,Y_test)
end = time.time()
print(F"训练集准确率：{train_score*100:8.2f}%，测试集准确率：{test_score*100:8.2f}%，Time:{end-start}s")

#------------------------------------------------
# 支持向量机
start = time.time()
classifier = SVC(gamma='auto',kernel='linear',C=1.0)
classifier.fit(X_train, Y_train)

train_score = classifier.score(X_train,Y_train)
test_score = classifier.score(X_test,Y_test)
end = time.time()
print(F"训练集准确率：{train_score*100:8.2f}%，测试集准确率：{test_score*100:8.2f}%，Time:{end-start}s")

# -----------------------------------------------
# 随机森林
start = time.time()
from sklearn.ensemble import RandomForestClassifier

rf_classifier = RandomForestClassifier(n_estimators=100)
rf_classifier.fit(X_train,Y_train)

train_score = rf_classifier.score(X_train,Y_train)
test_score = rf_classifier.score(X_test,Y_test)
end = time.time()
print(F"训练集准确率：{train_score*100:8.2f}%，测试集准确率：{test_score*100:8.2f}%，Time:{end-start}s")

# ==========================================
from sklearn.metrics import classification_report

# 首先计算预测值
pre_gau = gau_classifier.predict(X_test)
report_gau = classification_report(Y_test, pre_gau)
print(report_gau)

# ===========================================
# 其他分类器的分类报告


```

    数据加载完毕
    merge over
    feature label over
    feature age over
    feature workclass over
    feature occupation over
    feature counyrt over
    feature edu over
    feature status over
    feature fntwgt over
    feature edu-num over
    feature hours over
    feature gain over
    feature losss over
    feature race sex rel over
    feature creation  over
    encode over
    dataset split over
    训练集准确率：   80.33%，测试集准确率：   80.73%，Time:4.247833251953125s
    训练集准确率：   84.81%，测试集准确率：   73.36%，Time:1.7557365894317627s
    训练集准确率：   81.03%，测试集准确率：   81.11%，Time:1.4377596378326416s
    训练集准确率：   79.67%，测试集准确率：   79.88%，Time:1771.9325001239777s
    训练集准确率：   99.98%，测试集准确率：   84.98%，Time:5.411457061767578s
                  precision    recall  f1-score   support
    
               0       0.87      0.89      0.88     12435
               1       0.61      0.56      0.58      3846
    
        accuracy                           0.81     16281
       macro avg       0.74      0.72      0.73     16281
    weighted avg       0.81      0.81      0.81     16281
    
    


```python

```


```python

```
