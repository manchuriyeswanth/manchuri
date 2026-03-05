from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('app').getOrCreate()
csv_file = 'authors.csv'

file = spark.read.csv(csv_file, header=True)

file.printSchema() ## prints the schema aka type of valuea and headers 

file.show(5) # prints the first 5 values

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

## StructType and StructFiled are used to provide the schema information if the printSchema() displays wrong schema

schema = StructType([
    StructField(name="AUTHOR_ID",dataType=StringType(), nullable=True),
    StructField(name="AUTHOR_NAME",dataType=StringType(), nullable=True)
])

df = spark.read.csv(csv_file, header=True, schema=schema)

## 
df.show(10)
## SELECT select columns , here give the columns name to select specific columns 
selected_columns = df.select("AUTHOR_ID")
print(selected_columns.show(10))

## FILTER  - filter rows based on a condition
filtered_rows = df.filter(df.AUTHOR_ID>6) # Shows author id being greater than 6
print(filtered_rows.show(10))

##groupBy and aggregations
grouped_data = df.groupBy("AUTHOR_NAME").agg({"AUTHOR_ID":"avg"}) ## GroupBy columns and do a aggregation 

## SORT
sorted = df.orderBy("AUTHOR_ID")
print(sorted.show(10))
from pyspark.sql.functions import col,desc 
sorted_data = df.orderBy(col("AUTHOR_ID").desc(), col("AUTHOR_NAME").desc())
print(sorted_data.show(10))

## DISTINCT - Get Unique Rows

unique = df.select("AUTHOR_NAME").distinct()
print(unique.show(10))

## DROP - remove columns
dropped = df.drop("AUTHOR_ID")
print(dropped.show(10))

## WITH COLUMNS - Add calculated columns
from pyspark.sql.functions import concat, col, lit
add_columns = df.withColumn("AUTHOR__ID_NEW",concat(col("AUTHOR_ID"),lit("_"),col("AUTHOR_NAME"))) # use df. () notation to denote new column
print(add_columns.show(10))

## WithColumnRenamed - to rename columns for better readabaility

renamed_columns = add_columns.withColumnRenamed("AUTHOR__ID_NEW","NEW_ID")
print(renamed_columns.show(10))

## Spark DataFrme operatons
data_file = 'books.csv'
df2 = spark.read.csv(data_file,header=True,inferSchema=True)
print(df2.head(5))
print(df2.printSchema())

## select operation
selected_columns = df2.select('BOOK_ID','BOOK_TITLE','BOOK_COST','BOOK_EDITION')
print(selected_columns.head(5))

## Filter operation
# Filter

