
-- CSVConnectorDemo.
CREATE TABLE fs_source_table (
  user_id STRING,
  name STRING,
  age INT
  ,`file.path` STRING NOT NULL METADATA
  ,`file_name` STRING NOT NULL METADATA FROM 'file.name'
  ,`file_size` STRING NOT NULL METADATA FROM 'file.size'
  ,`file.modification-time` STRING NOT NULL METADATA
) PARTITIONED BY (age) WITH (
  'connector'='filesystem',
  'path'='file:///AI-Book/FlinkApplication/data/sql/connector/filesystem/csv/input'
  ,'format'='csv'
  ,'source.monitor-interval'='3'
);


CREATE TABLE fs_sink_table (
  user_id STRING,
  name STRING,
  age INT
  ,`file.path` STRING NOT NULL
  ,`file_name` STRING NOT NULL
  ,`file_size` STRING NOT NULL
  ,`file.modification-time` STRING NOT NULL
) PARTITIONED BY (age) WITH (
  'connector'='filesystem',
  'path'='file:///AI-Book/FlinkApplication/data/sql/connector/filesystem/csv/output'
  ,'format'='csv'
);

INSERT INTO fs_sink_table
SELECT
    user_id,
    name,
    age
    ,`file.path`
    ,`file_name`
    ,`file_size`
    ,`file.modification-time`
FROM fs_source_table;