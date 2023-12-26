# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)

# %%
df=pd.read_csv("loan_data_2007_2014.csv")

# %% [markdown]
# # Data Understanding

# %%
df.shape

# %%
df.info()

# %%
df.head()

# %% [markdown]
# ### Menentukan yang ingin di drop

# %%
# mengetahui apasaja column yang ada
df.columns

# %%
# menegecek adakah column yang hanya punya satu record unique
for column in df.columns:
  if len(df[column].unique()) == 1:
    print("\'" + column + "\',")

# %%
# menegecek adakah column yang punya unique setiap row
for column in df.columns:
  if len(df[column].unique()) == df.shape[0]:
    print("\'" + column + "\',")

# %%
# Mengecek column apa saja yang memiliki jumlah null nya adalah 80% dari jumlah seluruh row
for column in df.columns:
  if df[column].isna().sum() / len(df) > 0.8:
    print("\'" + column + "\',")

# %%
dropColumn = [
    # unique
    'id',
    'Unnamed: 0',
    
    
    # free text
    'url',
    
    # unique column
    'Unnamed: 0',
    'id',
    'member_id',
    'zip_code',
    
    # hanya satu record unique
    'policy_code',
    'application_type',
    'annual_inc_joint',
    'dti_joint',
    'verification_status_joint',
    'open_acc_6m',
    'open_il_6m',
    'open_il_12m',
    'open_il_24m',
    'mths_since_rcnt_il',
    'total_bal_il',
    'il_util',
    'open_rv_12m',
    'open_rv_24m',
    'max_bal_bc',
    'all_util',
    'inq_fi',
    'total_cu_tl',
    'inq_last_12m',
    
    # column apa saja yang memiliki jumlah null nya adalah 80% dari jumlah seluruh row
    'annual_inc_joint',
    'dti_joint',
    'verification_status_joint',
    'open_acc_6m',
    'open_il_6m',
    'open_il_12m',
    'open_il_24m',
    'mths_since_rcnt_il',
    'total_bal_il',
    'il_util',
    'open_rv_12m',
    'open_rv_24m',
    'max_bal_bc',
    'all_util',
    'inq_fi',
    'total_cu_tl',
    'inq_last_12m',
    'mths_since_last_record',
]

# %%
df = df.drop(dropColumn, axis = 1).copy()

# %%
df.shape

# %%
df.info()

# %%
df.describe()

# %% [markdown]
# ## Menuntukan label target

# %% [markdown]
# Column 'loan status' dipakai karena paling masuk akal untuk prediksi apakah seseorang akan mengembalikan pinjaman apa tidak

# %%
df['loan_status'].replace({
    'Fully Paid': 1,
    'Charged Off': 0,
    'Current': 1,
    'Late (31-120 days)': 0,
    'In Grace Period': 0,
    'Late (16-30 days)': 0,
    'Default': 0,
    'Does not meet the credit policy. Status:Fully Paid' : 1,
    'Does not meet the credit policy. Status:Charged Off' : 0,    
}, inplace = True)

# %%
df['loan_status'].value_counts()

# %%
df.isna().sum()

# %%
# tidak ada row yang duplicate
df.duplicated().sum()

# %% [markdown]
# ### Mengganti term, emp_length ke bentuk number

# %%
# buat atribut baru yang berisi integer dari term
df['term_month'] = df['term'].str.replace(r'months', r'')
df['emp_length'].unique()
df['emp_length_num'] = df['emp_length'].str.replace(r'\+ years', r'').str.replace(r'< 1 year', r'1').str.replace(r' year', r'').str.replace(r's', r'')

# %%
# DROP atribut yang sudah tidak terpakai
df = df.drop([
  'emp_length',
  'term',
  # free text
  'desc'
  ], axis=1).copy()

# %%
df.grade.value_counts()

# %% [markdown]
# ## Convert Type data

# %%
df['term_month'] = df['term_month'].astype('float32')
df['emp_length_num'] = df['emp_length_num'].astype('Int64')

# %% [markdown]
# ## Penanganan data dengan korelasi tinggi

# %%
sns.heatmap(df.corr())


# %% [markdown]
# bisa dilihat dari hetamap diatas ada beberapa atribut yang memiliki korelasi tinggi yang bisa menyebabkan overfitting maka perlu di drop, aku menggunakan 0.6 sebagai patokan

# %%
dfCorr = df.corr().abs()
columns = dfCorr.where(np.triu(np.ones(dfCorr.shape), k=1).astype(np.bool))
dropCorr = [column for column in columns.columns if any(columns[column] > 0.6)]

# %%
dropCorr

# %%
df.drop(dropCorr, axis=1, inplace=True)

# %% [markdown]
# ## Penaganan Data Date

# %% [markdown]
# earliest_cr_line dirubah ke integer dengan menghitung berapa hari menuju hari ini

# %%
df['earliest_cr_line'].head(3)

# %%
df['earliest_cr_line_day'] = pd.to_datetime(df['earliest_cr_line'], format='%b-%y')
df['earliest_cr_line_day'].head(3)

# %%
df['earliest_cr_line_month_now'] = round(pd.to_numeric((pd.Timestamp.now()  - df['earliest_cr_line_day']) / np.timedelta64(1, 'M')))
df['earliest_cr_line_month_now'].head(3)

# %%
df['earliest_cr_line_month_now'].describe()

# %%
df.drop(['earliest_cr_line', 'earliest_cr_line_day'], axis=1, inplace=True)

# %% [markdown]
# ### Issue_d

# %%
df['issue_d_day'] = pd.to_datetime(df['issue_d'], format='%b-%y')
df['issue_d_day_month_now'] = round(pd.to_numeric((pd.Timestamp.now()  - df['issue_d_day']) / np.timedelta64(1, 'M')))
df['issue_d_day_month_now'].head(3)

# %%
df.drop(['issue_d', 'issue_d_day'], axis=1, inplace=True)

# %% [markdown]
# ### last_credit_pull_d

# %%
df['last_credit_pull_d_day'] = pd.to_datetime(df['last_credit_pull_d'], format='%b-%y')
df['last_credit_pull_d_day_month_now'] = round(pd.to_numeric((pd.Timestamp.now()  - df['last_credit_pull_d_day']) / np.timedelta64(1, 'M')))
df['last_credit_pull_d_day_month_now'].head(3)

# %%
df.drop(['last_credit_pull_d', 'last_credit_pull_d_day'], axis=1, inplace=True)

# %% [markdown]
# ### last_pymnt_d

# %%
df['last_pymnt_d_day'] = pd.to_datetime(df['last_pymnt_d'], format='%b-%y')
df['last_pymnt_d_day_month_now'] = round(pd.to_numeric((pd.Timestamp.now()  - df['last_pymnt_d_day']) / np.timedelta64(1, 'M')))
df['last_pymnt_d_day_month_now'].head(3)

# %%
df.drop(['last_pymnt_d', 'last_pymnt_d_day'], axis=1, inplace=True)

# %% [markdown]
# ### next_pymnt_d

# %%
df['next_pymnt_d_day'] = pd.to_datetime(df['next_pymnt_d'], format='%b-%y')
df['next_pymnt_d_day_month_now'] = round(pd.to_numeric((pd.Timestamp.now()  - df['next_pymnt_d_day']) / np.timedelta64(1, 'M')))
df['next_pymnt_d_day_month_now'].head(3)

# %%
df.drop(['next_pymnt_d', 'next_pymnt_d_day'], axis=1, inplace=True)

# %% [markdown]
# ## Drop Value yang memiliki nilai unique yang banyak

# %%
df['sub_grade'].value_counts().count()
df.drop(['sub_grade'], axis=1, inplace=True)


# %%
df['title'].value_counts().count()
df.drop(['title'], axis=1, inplace=True)

# %%
df['emp_title'].value_counts().count()
df.drop(['emp_title'], axis=1, inplace=True)

# %%
df['dti'] = df['dti'].astype('int64')

# %%
df.drop('pymnt_plan', axis=1, inplace=True)

# %% [markdown]
# ## Mengisi NaN

# %%
df['annual_inc'].fillna(df['annual_inc'].mean(), inplace=True)
df['inq_last_6mths'].fillna(0, inplace=True)
df['pub_rec'].fillna(0, inplace=True)
df['open_acc'].fillna(0, inplace=True)
df['delinq_2yrs'].fillna(0, inplace=True)
df['acc_now_delinq'].fillna(0, inplace=True)
df['collections_12_mths_ex_med'].fillna(0, inplace=True)
df['tot_cur_bal'].fillna(0, inplace=True)
df['tot_coll_amt'].fillna(0, inplace=True)
df['emp_length_num'].fillna(0, inplace=True)
df['revol_util'].fillna(0, inplace=True)
df['mths_since_last_delinq'].fillna(-1, inplace=True)
df['next_pymnt_d_day_month_now'].fillna(-1, inplace=True)
df['last_pymnt_d_day_month_now'].fillna(-1, inplace=True)
df['last_credit_pull_d_day_month_now'].fillna(-1, inplace=True)
df['earliest_cr_line_month_now'].fillna(-1, inplace=True)

# %%
df.info()

# %%
df['emp_length_num'] = df['emp_length_num'].astype('int64')
df['term_month'] = df['term_month'].astype('float64')


# %%
categorical_cols = [col for col in df.select_dtypes(include='object').columns.tolist()]
onehot = pd.get_dummies(df[categorical_cols], drop_first=True)

# %%
numerical_cols = [col for col in df.columns.tolist() if col not in categorical_cols + ['loan_status']]
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
std = pd.DataFrame(ss.fit_transform(df[numerical_cols]), columns=numerical_cols)

# %%
data_model = pd.concat([onehot, std, df[['loan_status']]], axis=1)

# %%
from sklearn.model_selection import train_test_split

# %%
X = data_model.drop('loan_status', axis=1)
y = data_model['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
X_train.shape, X_test.shape

# %%
# Menghitung jumlah setiap nilai dalam kolom "Loan_Status"
loan_status_counts = y_train.value_counts()

# Membuat pie chart
plt.pie(loan_status_counts, labels=loan_status_counts.index, autopct='%1.1f%%', startangle=140, colors=['skyblue', 'lightgreen'])
plt.title('Distribusi Loan Status')
plt.show()

# %%
X_train.info()

# %% [markdown]
# ### Oversampling pada train_data

# %%
from imblearn.over_sampling import SMOTE 
sm = SMOTE(random_state = 42) 
X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel()) 

# %% [markdown]
# ### Naive bayes

# %%
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

clf = GaussianNB()
clf.fit(X_train_res, y_train_res)
Y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, Y_pred)

print("Akurasi {}".format(acc))
print(classification_report(y_test, Y_pred))
# print(list(le.classes_))

# %%
from sklearn.tree import DecisionTreeClassifier
tree_model = DecisionTreeClassifier(random_state=1)
tree_model.fit(X_train_res,y_train_res)
pred_cv_tree=tree_model.predict(X_test)
score_tree =accuracy_score(pred_cv_tree,y_test)*100 
print(score_tree)
print(classification_report(y_test, pred_cv_tree))

# %% [markdown]
# ## Radom Forest

# %%
from sklearn.ensemble import RandomForestClassifier
forest_model = RandomForestClassifier(random_state=1,max_depth=10,n_estimators=50)
forest_model.fit(X_train_res,y_train_res)
pred_cv_forest=forest_model.predict(X_test)
score_forest = accuracy_score(pred_cv_forest,y_test)*100
print("akurasi:",score_forest)
print(classification_report(y_test, pred_cv_forest))

# %% [markdown]
# ## XGBoost

# %%
from xgboost import XGBClassifier
xgb_model = XGBClassifier(n_estimators=50,max_depth=4)
xgb_model.fit(X_train_res,y_train_res)
pred_xgb=xgb_model.predict(X_test)
score_xgb = accuracy_score(pred_xgb,y_test)*100
print("Akurasi : ", score_xgb)
print(classification_report(y_test, pred_xgb))


