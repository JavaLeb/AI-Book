package flink.application.study.datastream.datasource.fileconnector;

import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.serialization.SimpleStringEncoder;
import org.apache.flink.connector.file.sink.FileSink;
import org.apache.flink.connector.file.src.FileSource;
import org.apache.flink.connector.file.src.reader.TextLineInputFormat;
import org.apache.flink.core.fs.Path;
import org.apache.flink.formats.json.JsonDeserializationSchema;
import org.apache.flink.formats.json.JsonSerializationSchema;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

import java.nio.charset.StandardCharsets;


public class JsonFormatDemo {
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        env.setParallelism(2);

        // 创建文件源
        Path inputPath = new Path("data/ds-api/file-connector/json/input");
        FileSource<String> fileSource = FileSource.forRecordStreamFormat(
                new TextLineInputFormat(),  // 按行读取文本
                inputPath // 文件路径
        ).build();

        DataStream<String> textStream = env.fromSource(
                fileSource,
                WatermarkStrategy.noWatermarks(),
                "text-source"
        );

        textStream.print("|--textStream-->");

        JsonDeserializationSchema<User> jsonFormat = new JsonDeserializationSchema<>(
                User.class);

        // 使用自定义反序列化器
        DataStream<User> userDS = textStream.map(jsonStr -> {
            try {
                jsonFormat.open(null); // 初始化mapper, 否则null.

                return jsonFormat.deserialize(jsonStr.getBytes());
            } catch (Exception e) {
                throw new RuntimeException("JSON解析失败: " + jsonStr, e);
            }
        });

        // 打印结果
        userDS.print("|--->");

        JsonSerializationSchema<User> serializationSchema = new JsonSerializationSchema<>();

        DataStream<String> userStrDS = userDS.map(new MapFunction<User, String>() {
            @Override
            public String map(User user) throws Exception {
                serializationSchema.open(null);
                return new String(serializationSchema.serialize(user), StandardCharsets.UTF_8);
            }
        });
        userStrDS.print("|--userStrDS-->");

        Path outPath = new Path("data/ds-api/file-connector/json/output");
        FileSink<String> fileSink = FileSink.forRowFormat(outPath, new SimpleStringEncoder<String>())
                .build();
        userStrDS.sinkTo(fileSink);

        env.execute();
    }
}
