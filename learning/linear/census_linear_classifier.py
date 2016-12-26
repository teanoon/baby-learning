import pandas
import tempfile
import tensorflow
import tensorflow.contrib.layers as layers
import tensorflow.contrib.learn.python.learn.estimators as estimators

COLUMNS = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
           'marital_status', 'occupation', 'relationship', 'race', 'gender',
           'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
           'income_bracket']
CATEGORICAL_COLUMNS = ['workclass', 'education', 'marital_status', 'occupation',
                       'relationship', 'race', 'gender', 'native_country']
CONTINUOUS_COLUMNS = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']


def read_data_sets_from_file(file, skiprows=0):
    datasets = pandas.read_csv(file, names=COLUMNS, skipinitialspace=True, skiprows=skiprows)
    datasets['label'] = (datasets['income_bracket'].apply(lambda x: '>50K' in x)).astype(int)
    continuous_columns = {key: tensorflow.constant(datasets[key].values) for key in CONTINUOUS_COLUMNS}
    categorical_columns = {key: tensorflow.SparseTensor(
        indices=[[index, 0] for index in range(datasets[key].size)],
        values=datasets[key].values,
        shape=[datasets[key].size, 1]) for key in CATEGORICAL_COLUMNS}
    feature_columns = dict(continuous_columns, **categorical_columns)
    label = tensorflow.constant(datasets['label'].values)
    return feature_columns, label


def build_model():
    # create categorical feature columns
    gender = layers.sparse_column_with_keys('gender', keys=['female', 'male'])
    workclass = layers.sparse_column_with_hash_bucket('workclass', hash_bucket_size=1000)
    education = layers.sparse_column_with_hash_bucket('education', hash_bucket_size=1000)
    marital_status = layers.sparse_column_with_hash_bucket('marital_status', hash_bucket_size=1000)
    occupation = layers.sparse_column_with_hash_bucket('occupation', hash_bucket_size=1000)
    relationship = layers.sparse_column_with_hash_bucket('relationship', hash_bucket_size=1000)
    race = layers.sparse_column_with_hash_bucket('race', hash_bucket_size=1000)
    native_country = layers.sparse_column_with_hash_bucket('native_country', hash_bucket_size=1000)

    # create continuous feature columns
    linear_age = layers.real_valued_column('age')
    age = layers.bucketized_column(linear_age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    education_num = layers.real_valued_column('education_num')
    capital_gain = layers.real_valued_column('capital_gain')
    capital_loss = layers.real_valued_column('capital_loss')
    hours_per_week = layers.real_valued_column('hours_per_week')

    # create crossed columns
    # the bucket size is decided by the number of crossed columns
    education_x_occupation = layers.crossed_column([education, occupation], hash_bucket_size=int(1e4))

    # create the model
    return estimators.LinearClassifier(
        feature_columns=[gender, workclass, education, marital_status, occupation, relationship, race, native_country,
                         linear_age, age, education_num, capital_gain, capital_loss, hours_per_week,
                         education_x_occupation],
        model_dir=tempfile.mkdtemp())


def load_training_set():
    return read_data_sets_from_file('resources/adult.data')


def load_test_data():
    return read_data_sets_from_file('resources/adult.test', skiprows=1)

# build model
model = build_model()

# train
model.fit(input_fn=load_training_set, steps=200)

# evaluate
results = model.evaluate(input_fn=load_test_data, steps=1)
print(results)
