package flink.application.study.datastream.datasource.fileconnector;

import org.apache.flink.connector.file.src.FileSourceSplit;
import org.apache.flink.connector.file.src.impl.StreamFormatAdapter;
import org.apache.flink.connector.file.src.reader.BulkFormat;
import org.apache.flink.connector.file.src.reader.TextLineInputFormat;
import org.apache.flink.formats.csv.CsvReaderFormat;

public class StreamFormat2BulkFormatDemo {
    public static void main(String[] args) {

        BulkFormat<String, FileSourceSplit> textBulkFormat =
                new StreamFormatAdapter<>(new TextLineInputFormat());

        BulkFormat<Pojo, FileSourceSplit> csvBulkFormat =
                new StreamFormatAdapter<>(CsvReaderFormat.forPojo(Pojo.class));

    }

    public static class Pojo{

    }
}
