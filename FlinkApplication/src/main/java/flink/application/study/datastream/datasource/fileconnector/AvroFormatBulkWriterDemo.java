package flink.application.study.datastream.datasource.fileconnector;

import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.connector.file.sink.FileSink;
import org.apache.flink.connector.file.src.FileSource;
import org.apache.flink.core.fs.Path;
import org.apache.flink.formats.avro.AvroWriters;
import org.apache.flink.formats.csv.CsvReaderFormat;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;


public class AvroFormatBulkWriterDemo {
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(2);


        // 定义CSV Format.
        CsvReaderFormat<User> csvFormat = CsvReaderFormat.forPojo(User.class);
        // 定义文件源.
        Path inputPath = new Path("data/ds-api/file-connector/avro/input/user.csv");
        FileSource<User> fileSource =
                FileSource.forRecordStreamFormat(csvFormat, inputPath).build();
        // 读取流.
        DataStream<User> inputDS =
                env.fromSource(fileSource, WatermarkStrategy.noWatermarks(), "csv-source");
        inputDS.print("|-csv-source->");

        Path outputBasePath = new Path("data/ds-api/file-connector/avro/output");

        final FileSink<User> bulkFormatSink = FileSink
                .forBulkFormat(outputBasePath, AvroWriters.forReflectRecord(User.class))
                .build();

        inputDS.sinkTo(bulkFormatSink);

        env.execute();
    }


}
