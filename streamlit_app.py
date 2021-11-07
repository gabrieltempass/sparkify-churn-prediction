import streamlit as st

import numpy as np
import pandas as pd

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, udf, from_unixtime, year, weekofyear, substring, encode, decode, split, desc, avg, first, concat_ws, countDistinct, sum as Fsum, max as Fmax, min as Fmin
from pyspark.sql.types import StringType, LongType, IntegerType, DateType, TimestampType
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler, PCA
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.pipeline import PipelineModel


def encode_decode_column(df, column, encoding, decoding):
    """Encode a column from a dataframe and then decode it.
    
    Parameters:
        df (pyspark.sql.dataframe.DataFrame): The dataframe that contains the
        	column.
        column (str): The name of the column to be encoded and decoded.
        encoding (str): The charset of the encoding (one of 'US-ASCII',
        	'ISO-8859-1', 'UTF-8', 'UTF-16BE', 'UTF-16LE', 'UTF-16').
        decoding (str): The charset of the decoding (one of 'US-ASCII',
        	'ISO-8859-1', 'UTF-8', 'UTF-16BE', 'UTF-16LE', 'UTF-16').
    
    Returns:
        df (pyspark.sql.dataframe.DataFrame): The dataframe with the column
        	properly encoded and decoded
        
    Example:
        df = encode_decode_column(df, 'column_name', 'ISO-8859-1', 'UTF-8')
    """
    
    df = df.withColumn(column, encode(column, encoding))
    df = df.withColumn(column, decode(column, decoding))
    
    return df


# Scala version implements .roc() and .pr()
# Python: https://spark.apache.org/docs/latest/api/python/_modules/pyspark/mllib/common.html
# Scala: https://spark.apache.org/docs/latest/api/java/org/apache/spark/mllib/evaluation/BinaryClassificationMetrics.html

class CurveMetrics(BinaryClassificationMetrics):
    """Put docstring here
    """
    
    def __init__(self, *args):
        super(CurveMetrics, self).__init__(*args)

    def _to_list(self, rdd):
        points = []
        # Note this collect could be inefficient for large datasets 
        # considering there may be one probability per datapoint (at most)
        # The Scala version takes a numBins parameter, 
        # but it doesn't seem possible to pass this from Python to Java
        for row in rdd.collect():
            # Results are returned as type scala.Tuple2, 
            # which doesn't appear to have a py4j mapping
            points += [(float(row._1()), float(row._2()))]
        return points

    def get_curve(self, method):
        rdd = getattr(self._java_model, method)().toJavaRDD()
        return self._to_list(rdd)


st.title('Sparkify Churn Prediction')
st.write('This is a web application, from a music streaming company, to \
	predict which users will churn. It receives a JSON file with the users \
	log as the input, processes the data and runs a machine learning model to \
	predict the churn probability for each user. Then, you can select a \
	probability range and download a file with the user IDs within that given \
	range.')

st.subheader('Step 1')
option = st.selectbox(
    label='What will be your input file?',
    options=('Select your file', 'mini_sparkify_event_data.json'))

if option == 'Select your file':
	st.stop()

# Read sparkify dataset
# Full dataset (12 Gb, 20 million rows)
# filepath = 's3n://udacity-dsnd/sparkify/sparkify_event_data.json'
# Mini dataset (128 Mb, 200 thousand rows, 1% of the full dataset)
# filepath = 's3n://udacity-dsnd/sparkify/mini_sparkify_event_data.json'

# Create spark session
spark = (SparkSession
    .builder
    .appName('Sparkify')
    .getOrCreate())

df_log = spark.read.json(option)
st.dataframe(df_log.limit(10).toPandas(), 1000, 600)

distinct_user_ids = df_log.dropDuplicates(['userId']).count()

st.write('Rows:', df_log.count())
st.write('Columns:', len(df_log.columns))
st.write('Distinct user IDs:', distinct_user_ids)

st.subheader('Step 2')
total = 1
n_steps = 32
percent_complete = total/n_steps
my_bar = st.progress(0.0)
my_bar.progress(percent_complete)

df_log = (df_log
    .withColumn('registration',
    			from_unixtime(col('registration')/1000).cast(TimestampType()))
    .withColumn('status',
    			col('status').cast(StringType()))
    .withColumn('ts',
    			from_unixtime(col('ts')/1000).cast(TimestampType()))
    .withColumn('userId',
    			col('userId').cast(LongType())))

percent_complete += total/n_steps
my_bar.progress(percent_complete)

df_log_valid = df_log.dropna(how='any', subset=['userId', 'sessionId'])

percent_complete += total/n_steps
my_bar.progress(percent_complete)

cat_cols = list(filter(lambda c: c[1] == 'string', df_log_valid.dtypes))
cat_cols = [item[0] for item in cat_cols]

percent_complete += total/n_steps
my_bar.progress(percent_complete)

# Fix the wrong encoding for the columns that are strings, and in order
# to retrieve the correct characters the encode-decode process must be
# done twice.
for column in cat_cols:
    df_log_valid = encode_decode_column(df_log_valid, column,
    									'ISO-8859-1', 'UTF-8')
    df_log_valid = encode_decode_column(df_log_valid, column,
    									'ISO-8859-1', 'UTF-8')

percent_complete += total/n_steps
my_bar.progress(percent_complete)

# Create the parsing functions with the user_agents library
get_browser     = udf(lambda x: parse(x).browser.family, StringType())
get_os          = udf(lambda x: parse(x).os.family, StringType())
get_device      = udf(lambda x: parse(x).device.family, StringType())
get_is_phone    = udf(lambda x: 1 if parse(x).is_mobile else 0, IntegerType())
get_is_tablet   = udf(lambda x: 1 if parse(x).is_tablet else 0, IntegerType())
get_is_computer = udf(lambda x: 1 if parse(x).is_pc else 0, IntegerType())

percent_complete += total/n_steps
my_bar.progress(percent_complete)

df_log_valid = (df_log_valid
    .withColumn('browser', get_browser('userAgent'))
    .withColumn('os', get_os('userAgent'))
    .withColumn('device', get_device('userAgent'))
    .withColumn('isPhone', get_is_phone('userAgent'))
    .withColumn('isTablet', get_is_tablet('userAgent'))
    .withColumn('isComputer', get_is_computer('userAgent')))

percent_complete += total/n_steps
my_bar.progress(percent_complete)

# Flag the churn
flag_cancellation_event = udf(
	lambda x: 1 if x == 'Cancellation Confirmation' else 0,
	IntegerType())
df_log_valid = (df_log_valid
	.withColumn('churn', flag_cancellation_event('page')))
window_val = (Window
	.partitionBy('userId')
	.orderBy(desc('ts'))
	.rangeBetween(Window.unboundedPreceding, Window.currentRow))
df_log_valid = (df_log_valid
	.withColumn('churned', Fsum('churn')
	.over(window_val)))

percent_complete += total/n_steps
my_bar.progress(percent_complete)

# Add the cohort
df_log_valid = (df_log_valid
	.withColumn('cohort', substring('registration', 1, 7)))

percent_complete += total/n_steps
my_bar.progress(percent_complete)

# Create a user feature matrix
df_users_features = (df_log_valid
	.select('userId', 'churned')
	.dropDuplicates(['userId']))

percent_complete += total/n_steps
my_bar.progress(percent_complete)

# Create year and week columns to aggregate data in the next steps
df_log_valid = (df_log_valid
    .withColumn('year', year(col('ts').cast(DateType())))
    .withColumn('week', weekofyear(col('ts').cast(DateType())))
    .withColumn('yearWeek', concat_ws('-', col('year'), col('week'))))

percent_complete += total/n_steps
my_bar.progress(percent_complete)

# Add the avg. songs per week
df_avg_songs_week = (df_log_valid
    .where('page = "NextSong"')
    .groupby('userId','yearWeek')
    .count()
    .groupBy('userId')
    .agg(avg('count').alias('avgSongsWeek')))

percent_complete += total/n_steps
my_bar.progress(percent_complete)

# Add the avg. sessions per week
df_avg_sessions_week = (df_log_valid
    .groupby('userId','yearWeek')
    .agg(countDistinct('sessionId').alias('sessions'))
    .groupBy('userId')
    .agg(avg('sessions').alias('avgSessionsWeek')))

percent_complete += total/n_steps
my_bar.progress(percent_complete)

# Add the avg. session duration
df_avg_session_duration = (df_log_valid
    .groupby('userId','sessionId')
    .agg(Fmin('ts').alias('start'), Fmax('ts').alias('end'))
    .withColumn('sessionDuration', col('end').cast(LongType()) - col('start').cast(LongType()))
    .groupBy('userId')
    .agg(avg('sessionDuration').alias('avgSessionDuration')))

percent_complete += total/n_steps
my_bar.progress(percent_complete)

# Add the cohort
df_cohort = (df_log_valid
    .withColumn('cohort', substring('registration', 1, 7))
    .select('userId', 'cohort')
    .dropDuplicates(['userId']))

percent_complete += total/n_steps
my_bar.progress(percent_complete)

# Add the average song length
df_length = (df_log_valid
    .groupBy('userId')
    .agg(avg('length').alias('length')))

percent_complete += total/n_steps
my_bar.progress(percent_complete)

# Add the most recent metropolitan area
df_metro_area = (df_log_valid
    .withColumn('metropolitanArea', split('location', ',')[0])
    .orderBy(desc('ts'))
    .groupBy('userId')
    .agg(first('metropolitanArea').alias('metropolitanArea')))

percent_complete += total/n_steps
my_bar.progress(percent_complete)

# Add the most recent state
df_state = (df_log_valid
    .withColumn('state', split('location', ',')[1])
    .orderBy(desc('ts'))
    .groupBy('userId')
    .agg(first('state').alias('state')))

percent_complete += total/n_steps
my_bar.progress(percent_complete)

# Add the most recent gender
df_gender = (df_log_valid
    .orderBy(desc('ts'))
    .groupBy('userId')
    .agg(first('gender').alias('gender')))

percent_complete += total/n_steps
my_bar.progress(percent_complete)

# Add the most recent level
df_level = (df_log_valid
    .orderBy(desc('ts'))
    .groupBy('userId')
    .agg(first('level').alias('level')))

percent_complete += total/n_steps
my_bar.progress(percent_complete)

# Add the most used browser
df_browser = (df_log_valid
    .select('userId', 'browser')
    .groupBy('userId', 'browser')
    .count()
    .orderBy(desc('count'))
    .groupBy('userId')
    .agg(first('browser').alias('browser')))

percent_complete += total/n_steps
my_bar.progress(percent_complete)

# Add the most used OS
df_os = (df_log_valid
    .select('userId', 'os')
    .groupBy('userId', 'os')
    .count()
    .orderBy(desc('count'))
    .groupBy('userId')
    .agg(first('os').alias('os')))

percent_complete += total/n_steps
my_bar.progress(percent_complete)

# Add the most used device
df_device = (df_log_valid
    .select('userId', 'device')
    .groupBy('userId', 'device')
    .count()
    .orderBy(desc('count'))
    .groupBy('userId')
    .agg(first('device').alias('device')))

percent_complete += total/n_steps
my_bar.progress(percent_complete)

# Add the most used is phone
df_is_phone = (df_log_valid
    .select('userId', 'isPhone')
    .groupBy('userId', 'isPhone')
    .count()
    .orderBy(desc('count'))
    .groupBy('userId')
    .agg(first('isPhone').alias('isPhone')))

percent_complete += total/n_steps
my_bar.progress(percent_complete)

# Add the most used is tablet
df_is_tablet = (df_log_valid
    .select('userId', 'isTablet')
    .groupBy('userId', 'isTablet')
    .count()
    .orderBy(desc('count'))
    .groupBy('userId')
    .agg(first('isTablet').alias('isTablet')))

percent_complete += total/n_steps
my_bar.progress(percent_complete)

# Add the most used is computer
df_is_computer = (df_log_valid
    .select('userId', 'isComputer')
    .groupBy('userId', 'isComputer')
    .count()
    .orderBy(desc('count'))
    .groupBy('userId')
    .agg(first('isComputer').alias('isComputer')))

percent_complete += total/n_steps
my_bar.progress(percent_complete)

df_users_features = (df_users_features
    .join(df_avg_songs_week, on='userId')
    .join(df_avg_sessions_week, on='userId')
    .join(df_avg_session_duration, on='userId')
    .join(df_cohort, on='userId')
    .join(df_length, on='userId')
    .join(df_metro_area, on='userId')
    .join(df_state, on='userId')
    .join(df_gender, on='userId')
    .join(df_level, on='userId')
    .join(df_browser, on='userId')
    .join(df_os, on='userId')
    .join(df_device, on='userId')
    .join(df_is_phone, on='userId')
    .join(df_is_tablet, on='userId')
    .join(df_is_computer, on='userId'))

percent_complete += total/n_steps
my_bar.progress(percent_complete)

feature_vector    = 'features'
target_vector     = 'label'
prediction_vector = 'prediction'

df_processed = (df_users_features
    .drop('userId', 'metropolitanArea', 'state')
    .withColumnRenamed('churned', target_vector))

percent_complete += total/n_steps
my_bar.progress(percent_complete)

indexer_level   = StringIndexer(inputCol='level',   outputCol='levelIndex')
indexer_gender  = StringIndexer(inputCol='gender',  outputCol='genderIndex')
indexer_cohort  = StringIndexer(inputCol='cohort',  outputCol='cohortIndex')
indexer_browser = StringIndexer(inputCol='browser', outputCol='browserIndex')
indexer_device  = StringIndexer(inputCol='device',  outputCol='deviceIndex')
indexer_os      = StringIndexer(inputCol='os',      outputCol='osIndex')

percent_complete += total/n_steps
my_bar.progress(percent_complete)

ohe_inputs  = ['levelIndex', 'genderIndex', 'cohortIndex', 'browserIndex', 'deviceIndex', 'osIndex']
ohe_outputs = ['levelOhe',   'genderOhe',   'cohortOhe',   'browserOhe',   'deviceOhe',   'osOhe']

one_hot_encoder = OneHotEncoderEstimator(inputCols=ohe_inputs, outputCols=ohe_outputs)

percent_complete += total/n_steps
my_bar.progress(percent_complete)

va_inputs = ['avgSongsWeek',
             'avgSessionsWeek',
             'avgSessionDuration',
             'length',
             'isPhone',
             'isTablet',
             'isComputer',
             'levelOhe',
             'genderOhe',
             'cohortOhe',
             'browserOhe',
             'deviceOhe',
             'osOhe']

vector_assembler = VectorAssembler(inputCols=va_inputs, outputCol=feature_vector)

pipeline_process = Pipeline(stages=[
    indexer_level,
    indexer_gender,
    indexer_cohort,
    indexer_browser,
    indexer_device,
    indexer_os,
    one_hot_encoder,
    vector_assembler
])

percent_complete += total/n_steps
my_bar.progress(percent_complete)

# df_processed = pipeline_process.fit(df_processed).transform(df_processed)

seed = 0
df_train, df_test = df_processed.randomSplit((0.8, 0.2), seed=seed)

percent_complete += total/n_steps
my_bar.progress(1.0)

# To load the trained model:
path = '/Users/gabriel.tempass/Repositories/sparkify-churn-prediction/model'
best_model = PipelineModel.load(path)
predictions = best_model.transform(df_test)


# lr = LogisticRegression(featuresCol=feature_vector,
#                         labelCol=target_vector,
#                         predictionCol=prediction_vector)

# metric_auc = 'areaUnderROC'
# metric_ac = 'accuracy'
# metric_f1 = 'f1'

# evaluator_auc = BinaryClassificationEvaluator(metricName=metric_auc)
# evaluator_ac = MulticlassClassificationEvaluator(predictionCol=prediction_vector,
#                                                  labelCol=target_vector,
#                                                  metricName=metric_ac)
# evaluator_f1 = MulticlassClassificationEvaluator(predictionCol=prediction_vector,
#                                                  labelCol=target_vector,
#                                                  metricName=metric_f1)

# lr_e = Pipeline(stages=[lr])
# b_grid = ParamGridBuilder().addGrid(lr.maxIter, [1]).build()
# b_cv = CrossValidator(estimator=lr_e,
#                       estimatorParamMaps=b_grid,
#                       evaluator=evaluator_auc,
#                       numFolds=3,
#                       seed=seed)

# b_model = b_cv.fit(df_train)

# b_predictions = b_model.transform(df_test)

# label_mean = b_predictions.groupBy().mean('label').collect()[0][0]
# null_ac = max(label_mean, 1 - label_mean)

# auc = evaluator_auc.evaluate(b_predictions)
# ac = evaluator_ac.evaluate(b_predictions)
# f1 = evaluator_f1.evaluate(b_predictions)

# b_best_model = b_model.bestModel

# st.write('The AUC score on the test set is: {:.4%}'.format(auc))
# st.write('The null accuracy score on the test set is: {:.4%}'.format(null_ac))
# st.write('The accuracy score on the test set is: {:.4%}'.format(ac))
# st.write('The F1 score on the test set is: {:.4%}'.format(f1))