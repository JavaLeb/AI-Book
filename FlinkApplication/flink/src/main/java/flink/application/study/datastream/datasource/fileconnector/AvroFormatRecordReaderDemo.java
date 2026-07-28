package flink.application.study.datastream.datasource.fileconnector;


import org.apache.flink.core.fs.Path;
import org.apache.flink.formats.avro.AvroInputFormat;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

/**
 * 逐记录 读取 Avro格式文件 演示.
 */
public class AvroFormatRecordReaderDemo {
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(2);

        Path inputPath = new Path("data/ds-api/file-connector/avro/input/user.avro");
        AvroInputFormat<User> users = new AvroInputFormat<>(inputPath, User.class);
        DataStream<User> usersDS = env.createInput(users);
        usersDS.print("|-->");

        env.execute();
    }
}
