# from pyspark.sql import SparkSession

# spark = SparkSession.builder.appName('get started').getOrCreate()

# data = {("alice", 25),("bob", 34),("charlie",46)}

# df = spark.createDataFrame(data,["Name", "Age"])

# df.show()

## Spark Context vs Spark Session

## Spark Context -> Connection to Spark Cluster and mamages job execution across cluster , Entry point in 1.x

## Spark Sesison -> Combines functionalities of Spark Context, SQL Context, Hive Context, Strwaming Context, Multiple programming Lanhuages, unified entry point 
## in 2.x

## Spark Context -> Lower level programming, create RDD, make transformations and actions on data
## Spark Session -> Extend Spark Context and provide higher level abstractions like Dataframes using dataframe api, and other programming libraries

## Spark Session offers unified entry point acting as context too. 

## 

# from pyspark.sql import SparkSession

# spark = SparkSession.builder.appName("my-app").config("spark.executor.memory",'1g').getOrCreate()

# spark

# spark.stop()

## RDD - resilient distributed datasets

## RDD Operations 

## Transformations - Create new RDD or apply some manipulation on existing RDD, Lazy evaluation build a lineage graph (not immediately executed)
## Some of them include - Map, filter, flatmap, reducebykey, sortby, join

## Actions - Retrieve result or action on RDD, eager evaluation / data movement or computation
## Collect, count, first, take, save , for each

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("RDD - DEMO").getOrCreate()
numbers = [1,2,3,4,5]

rdd = spark.sparkContext.parallelize(numbers)
## Collect Action 
print(rdd.collect())

## RDD Actions
data = [("alice", 35),("Bob", 24)]

rdd = spark.sparkContext.parallelize(data)

print("All Elements of data",rdd.collect())

print("Element Count:",rdd.count())

print("First lement:",rdd.first())

print("Take 2 elements",rdd.take(2))  # parameter being sent as 2

## rdd.collect()
## rdd.count()
## rdd.first()
## rdd.take(parameter num)
## rdd.foreach(lambda functions)
rdd.foreach(lambda x: print(x))

## RDD Transformations
#MAP
mapped_rdd = rdd.map(lambda x: (x[0].upper(), x[1]))
print(mapped_rdd.collect())

# Filter 
filtered_rdd = rdd.filter(lambda x : x[1]>30)
print(filtered_rdd.collect())

## Reducebykey applies a function to the matching keys 

reduced_rdd = rdd.reduceByKey(lambda x,y: x+y)
print(reduced_rdd.collect())

## SortBy - Sort by transformation ti get values in desc orger 
sorted_rdd = rdd.sortBy(lambda x:x[1], ascending =False)
print(sorted_rdd.collect())

## Saving to text file and reading ferom text file
## Each element writeen as string and on a new file
#rdd.saveAsTextFile("output.txt")

## Reading 
rdd_text = spark.sparkContext.textFile('output.txt')
print("RDD Reading from txt file")
print(rdd_text.collect())

##