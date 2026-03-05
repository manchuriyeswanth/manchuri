## Data frame power abstravtons to work with tabular data in spark. 
## They are API 
## Like a table in relational database in rows and columns
## Schema information to optimize verious queries 

## Adv of DF's over RDD
# Schema information
# SQL Like interface
# Integrate with the Spark Eco System

from pyspark.sql import SparkSession
from pyspark.sql.functions import desc

spark = SparkSession.builder.appName('totuo2').getOrCreate()

rdd = spark.sparkContext.textFile('authors.csv')

result_rdd = rdd.flatMap(lambda line: line.split(" ")).map(lambda word:(word,1)).reduceByKey(lambda a,b: a+b).sortBy(lambda x: x[1], ascending = False)

##print(result_rdd.collect())
## Flatmap reduces the sentence by flattening out 
## Then creating dictionary
## reduceByKey to take the arguments 
## sortBy to sort 

df = spark.read.csv("authors.csv", header=True)

df.print_schema()
result_df = df.selectExpr("explode(split(value,' ')) as word").groupBy("word").count().orderBy(desc("count"))

print(result_df.take(10))



