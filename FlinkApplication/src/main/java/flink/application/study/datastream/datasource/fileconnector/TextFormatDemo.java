package flink.application.study.datastream.datasource.fileconnector;


import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.serialization.BulkWriter;
import org.apache.flink.api.common.serialization.SimpleStringEncoder;
import org.apache.flink.connector.file.sink.FileSink;
import org.apache.flink.connector.file.src.FileSource;
import org.apache.flink.connector.file.src.FileSourceSplit;
import org.apache.flink.connector.file.src.impl.StreamFormatAdapter;
import org.apache.flink.connector.file.src.reader.BulkFormat;
import org.apache.flink.connector.file.src.reader.TextLineInputFormat;
import org.apache.flink.core.fs.FSDataOutputStream;
import org.apache.flink.core.fs.Path;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.time.Duration;


public class TextFormatDemo {
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(2);

        // 读取目录.
        Path boundedPath = new Path("data/ds-api/connector/text/bounded");

        // 逐行读取为有界流.
        FileSource<String> recordBoundedFileSource = FileSource.forRecordStreamFormat(
                new TextLineInputFormat(), boundedPath).build();
        DataStream<String> recordBoundedDS =
                env.fromSource(recordBoundedFileSource, WatermarkStrategy.noWatermarks(), "record-bounded-file-source");
        recordBoundedDS.print("|-record-bounded-file-source->");

        // 批量读取为有界流.
        // StreamFormat -> BulkFormat.
        BulkFormat<String, FileSourceSplit> bulkFormat = new StreamFormatAdapter<>(new TextLineInputFormat());
        // 批量读取.
        FileSource<String> bulkBoundedFileSource = FileSource.forBulkFileFormat(bulkFormat, boundedPath).build();
        // 有界流.
        DataStream<String> bulkBoundedDS =
                env.fromSource(bulkBoundedFileSource, WatermarkStrategy.noWatermarks(), "bulk-bounded-file-source");
        bulkBoundedDS.print("|-bulk-bounded-file-source->");

        // 读取目录.
        Path continuousPath = new Path("data/ds-api/connector/text/continuous");

        // 逐行读取为无界流.
        FileSource<String> recordContinuousFilesource = FileSource.forRecordStreamFormat(
                        new TextLineInputFormat(), continuousPath)
                .monitorContinuously(Duration.ofSeconds(1L))
                .build();
        DataStream<String> recordContinuousDS =
                env.fromSource(recordContinuousFilesource, WatermarkStrategy.noWatermarks(),
                        "record-continuous-file-source");
        recordContinuousDS.print("|-record-continuous-file-source->");

        // 批量读取为无界流.
        FileSource<String> bulkContinuousFilesource = FileSource.forBulkFileFormat(
                        bulkFormat, continuousPath)
                .monitorContinuously(Duration.ofSeconds(1L))
                .build();
        DataStream<String> bulkContinuousDS =
                env.fromSource(bulkContinuousFilesource, WatermarkStrategy.noWatermarks(),
                        "bulk-continuous-file-source");
        bulkContinuousDS.print("|-bulk-continuous-file-source->");

        // 写出目录.
        Path record2RowOutPath = new Path("data/ds-api/connector/text/output/record2row");
        Path record2BulkOutpath = new Path("data/ds-api/connector/text/output/record2Bulk");
        Path bulk2RowOutpath = new Path("data/ds-api/connector/text/output/bulk2Row");
        Path bulk2BulkOutpath = new Path("data/ds-api/connector/text/output/bulk2Bulk");

        FileSink<String> record2RowFileSink = FileSink.forRowFormat(
                record2RowOutPath, new SimpleStringEncoder<String>()).build();

        FileSink<String> record2BulkFileSink = FileSink.forBulkFormat(
                record2BulkOutpath, new TextBulkWriterFactory<String>()).build();

        FileSink<String> bulk2RowFileSink = FileSink.forRowFormat(
                bulk2RowOutpath, new SimpleStringEncoder<String>()).build();

        FileSink<String> bulk2BulkFileSink = FileSink.forBulkFormat(
                bulk2BulkOutpath, new TextBulkWriterFactory<String>()).build();

        recordContinuousDS.sinkTo(record2RowFileSink);
        recordContinuousDS.sinkTo(record2BulkFileSink);
        bulkContinuousDS.sinkTo(bulk2BulkFileSink);

        env.execute();
    }

    public static class TextBulkWriterFactory<T> implements BulkWriter.Factory<T> {
        @Override
        public BulkWriter<T> create(FSDataOutputStream out) throws IOException {
            return new TextBulkWriter<>(out);
        }
    }

    public static class TextBulkWriter<T> implements BulkWriter<T> {
        private final FSDataOutputStream outputStream;
        private final StringBuilder buffer = new StringBuilder();
        private static final Integer BUFFER_SIZE = 1024 * 1024;

        public TextBulkWriter(FSDataOutputStream outputStream) {
            this.outputStream = outputStream;
        }

        @Override
        public void addElement(T element) throws IOException {
            buffer.append(element.toString()).append("\n"); // 添加换行符
            if (buffer.length() >= BUFFER_SIZE) { // 按 1MB 批量写入
                flush();
            }
        }

        @Override
        public void flush() throws IOException {
            outputStream.write(buffer.toString().getBytes(StandardCharsets.UTF_8));
            buffer.setLength(0); // 清空缓冲区
        }

        @Override
        public void finish() throws IOException {
            flush(); // 确保最后数据写入
        }
    }


}
