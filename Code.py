from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
import time
from pyspark.ml.classification import LinearSVC
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from pyspark.shell import sqlContext
from pyspark.sql.types import IntegerType

spark_UI = SparkSession.builder.appName('BDA_Project_AIE').getOrCreate()
df_ids=spark_UI.read.format("csv").option("header","true").option("mode","PERMISSIVE").option("mode","DROPMALFORMED").load("/media/sf_VBoxSharedFolder/Webpages_Classification_test_data.csv") 
df_ids = df_ids.withColumn("js_len", df_ids["js_len"].cast(IntegerType()))
df_ids = df_ids.withColumn("js_obf_len", df_ids["js_obf_len"].cast(IntegerType()))
df_ids=df_ids.na.drop()
good=df_ids.filter((df_ids.label=='good'))
bad=df_ids.filter((df_ids.label=='bad'))
df_ids=good.union(bad)
column_set=df_ids.columns
tokenizer = Tokenizer(outputCol="words")
tokenizer.setInputCol("content")
remover = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol="contentVec")
hashingTF = HashingTF(inputCol=remover.getOutputCol(), outputCol="rawFeatures", numFeatures=20)
regexTokenizer = RegexTokenizer(inputCol="url", outputCol="urlToken", pattern="/")
hashingTFurl = HashingTF(inputCol=regexTokenizer.getOutputCol(), outputCol="urlVec", numFeatures=20)
indexertld = StringIndexer(inputCol="tld", outputCol="tldIndex")
indexerlabels = StringIndexer(inputCol="label", outputCol="labelIndex")
indexerhttps= StringIndexer(inputCol="https", outputCol="httpsIndex")
indexerwhois= StringIndexer(inputCol="who_is", outputCol="whoisIndex")
Encodertld = OneHotEncoder(inputCols=[indexertld.getOutputCol()], outputCols=["tldVec"])
Encoderhttps = OneHotEncoder(inputCols=[indexerhttps.getOutputCol()], outputCols=["httpsVec"])
Encoderwhois = OneHotEncoder(inputCols=[indexerwhois.getOutputCol()], outputCols=["whoisVec"])
assembler = VectorAssembler(inputCols=["rawFeatures", "urlVec", "tldVec","httpsVec","whoisVec","js_len","js_obf_len"], outputCol="features")
pipeline = Pipeline(stages = [indexerlabels,tokenizer,remover,hashingTF,regexTokenizer,hashingTFurl,indexertld,indexerhttps,indexerwhois,Encodertld,Encoderhttps,Encoderwhois,assembler])
pipelineModel = pipeline.fit(df_ids)
df_ids = pipelineModel.transform(df_ids)
selected_Columns = ['labelIndex', 'features'] + column_set
df_ids = df_ids.select(selected_Columns)
training_data, testing_data = df_ids.randomSplit([0.6805,0.3195], seed = 99999999)
print("Training Dataset Count: " + str(training_data.count()))
print("Test Dataset Count: " + str(testing_data.count()))
NaiveBayes = NaiveBayes(labelCol="labelIndex", featuresCol="features", smoothing=1.0, modelType="multinomial" )

start=time.time()
NaiveBayesModel = NaiveBayes.fit(training_data)
end=time.time()
start1=time.time()
f_predictions = NaiveBayesModel.transform(testing_data)
f_predictions.select('labelIndex','prediction').coalesce(1).write.option("header","true").csv('pred_output.csv')
end1=time.time()

print("Time taken for training:")
print(end-start)
print("Time taken for predicting:")
print(end1-start1)




evaluation1 = BinaryClassificationEvaluator(labelCol="labelIndex",rawPredictionCol="prediction")
print("Accuracy of model")
print(evaluation1.evaluate(f_predictions))


evaluation2 =MulticlassClassificationEvaluator(labelCol="labelIndex",predictionCol="prediction", metricName="accuracy")
accuracy = evaluation2.evaluate(f_predictions)
print("ACCURACY :")
print(accuracy)


evaluation3 =MulticlassClassificationEvaluator(labelCol="labelIndex",predictionCol="prediction", metricName="f1")
score = evaluation3.evaluate(f_predictions)
print("F1-score :")
print(score)




Linearsvc = LinearSVC(labelCol="labelIndex", featuresCol="features", maxIter=100)

start=time.time()
LinearSVCModel = Linearsvc.fit(training_data)
end=time.time()
start1=time.time()
f_predictions3 = LinearSVCModel.transform(testing_data)
end1=time.time()



print("Time to train:")
print(end-start)
print("Time to predict:")
print(end1-start1)

evaluation4 = BinaryClassificationEvaluator(labelCol="labelIndex",rawPredictionCol="prediction")
print("Accuracy of model")
print(evaluation4.evaluate(f_predictions3))


evaluation5 =MulticlassClassificationEvaluator(labelCol="labelIndex",predictionCol="prediction", metricName="accuracy")
accuracy = evaluation5.evaluate(f_predictions3)
print("ACCURACY :")
print(accuracy)


evaluation6 =MulticlassClassificationEvaluator(labelCol="labelIndex",predictionCol="prediction", metricName="f1")
score = evaluation6.evaluate(f_predictions3)
print("F1-score :")
print(score)

