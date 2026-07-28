package flink.application.study.datastream.datasource.fileconnector;


import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.serialization.BulkWriter;
import org.apache.flink.api.common.serialization.SimpleStringEncoder;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.connector.file.sink.FileSink;
import org.apache.flink.connector.file.src.FileSource;
import org.apache.flink.core.fs.FSDataOutputStream;
import org.apache.flink.core.fs.Path;
import org.apache.flink.formats.common.Converter;
import org.apache.flink.formats.csv.CsvReaderFormat;
import org.apache.flink.shaded.jackson2.com.fasterxml.jackson.annotation.JsonProperty;
import org.apache.flink.shaded.jackson2.com.fasterxml.jackson.core.JsonEncoding;
import org.apache.flink.shaded.jackson2.com.fasterxml.jackson.core.JsonGenerator;
import org.apache.flink.shaded.jackson2.com.fasterxml.jackson.databind.SerializationFeature;
import org.apache.flink.shaded.jackson2.com.fasterxml.jackson.dataformat.csv.CsvMapper;
import org.apache.flink.shaded.jackson2.com.fasterxml.jackson.dataformat.csv.CsvSchema;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.util.FlinkRuntimeException;
import org.apache.flink.util.function.SerializableFunction;
import org.apache.flink.util.jackson.JacksonMapperFactory;

import javax.annotation.Nullable;
import java.io.IOException;
import java.math.BigDecimal;
import java.time.Duration;

import static org.apache.flink.util.Preconditions.checkNotNull;

public class CSVFormatDemo {
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(2);

        // 定义 csvMapper -> csvSchema 映射的函数.
        SerializableFunction<CsvMapper, CsvSchema> schemaGenerator = csvMapper ->
                csvMapper.schemaFor(City.class).withoutQuoteChar().withColumnSeparator('|');

        // 定义CSV Format.
        CsvReaderFormat<City> csvFormat =
                CsvReaderFormat.forSchema(CsvMapper::new, schemaGenerator, TypeInformation.of(City.class));

        // 定义文件源.
        FileSource<City> fileSource =
                FileSource.forRecordStreamFormat(csvFormat,
                                new Path("data/ds-api/file-connector/csv/input"))
                        .monitorContinuously(Duration.ofSeconds(1L))
                        .build();

        final DataStream<City> continuousDS =
                env.fromSource(fileSource, WatermarkStrategy.noWatermarks(), "continuous-file-source");
        continuousDS.print("|-continuous-file-source->");

        Path outputPath = new Path("data/ds-api/file-connector/csv/output");

        // 将数据作为字符串写入文件.
        FileSink<City> rowSink = FileSink.forRowFormat(outputPath,
                new SimpleStringEncoder<City>()).build();

        FileSink<City> bulkSink = FileSink.forBulkFormat(outputPath,
                new CsvBulkWriterFactory()).build();

        continuousDS.sinkTo(rowSink);

        // 文件内容为空, 需要修复bug.
        continuousDS.sinkTo(bulkSink);

        env.execute();
    }

    public static class People {

    }

    public static class City {
        @JsonProperty(index = 1)
        public String city;
        @JsonProperty(index = 2)
        public BigDecimal lat;
        @JsonProperty(index = 3)
        public BigDecimal lng;
        @JsonProperty(index = 4)
        public String country;
        @JsonProperty(index = 5)
        public String iso2;
        @JsonProperty(index = 6)
        public String adminName;
        @JsonProperty(index = 7)
        public String capital;
        @JsonProperty(index = 8)
        public long population;

        public String getCity() {
            return city;
        }

        public void setCity(String city) {
            this.city = city;
        }

        public BigDecimal getLat() {
            return lat;
        }

        public void setLat(BigDecimal lat) {
            this.lat = lat;
        }

        public BigDecimal getLng() {
            return lng;
        }

        public void setLng(BigDecimal lng) {
            this.lng = lng;
        }

        public String getCountry() {
            return country;
        }

        public void setCountry(String country) {
            this.country = country;
        }

        public String getIso2() {
            return iso2;
        }

        public void setIso2(String iso2) {
            this.iso2 = iso2;
        }

        public String getAdminName() {
            return adminName;
        }

        public void setAdminName(String adminName) {
            this.adminName = adminName;
        }

        public String getCapital() {
            return capital;
        }

        public void setCapital(String capital) {
            this.capital = capital;
        }

        public long getPopulation() {
            return population;
        }

        public void setPopulation(long population) {
            this.population = population;
        }

        @Override
        public String toString() {
            return "City{" +
                    "city='" + city + '\'' +
                    ", lat=" + lat +
                    ", lng=" + lng +
                    ", country='" + country + '\'' +
                    ", iso2='" + iso2 + '\'' +
                    ", adminName='" + adminName + '\'' +
                    ", capital='" + capital + '\'' +
                    ", population=" + population +
                    '}';
        }
    }


    public static class CsvBulkWriterFactory implements BulkWriter.Factory<City> {
        @Override
        public BulkWriter<City> create(FSDataOutputStream out) throws IOException {
            return CsvBulkWriter.forPojo(City.class, out);
        }
    }

    public static class CsvBulkWriter<T, R, C> implements BulkWriter<T> {

        private final FSDataOutputStream stream;
        private final Converter<T, R, C> converter;
        @Nullable
        private final C converterContext;
        private final JsonGenerator generator;

        CsvBulkWriter(
                CsvMapper mapper,
                CsvSchema schema,
                Converter<T, R, C> converter,
                @Nullable C converterContext,
                FSDataOutputStream stream) {
            checkNotNull(mapper);
            checkNotNull(schema);

            // Prevent Jackson's writeValue() method calls from closing the stream.
            mapper.getFactory().disable(JsonGenerator.Feature.AUTO_CLOSE_TARGET);
            mapper.disable(SerializationFeature.FLUSH_AFTER_WRITE_VALUE);

            this.converter = checkNotNull(converter);
            this.stream = checkNotNull(stream);
            this.converterContext = converterContext;
            try {
                this.generator = mapper.writer(schema).createGenerator(stream, JsonEncoding.UTF8);
            } catch (IOException e) {
                throw new FlinkRuntimeException("Could not create CSV generator.", e);
            }
        }

        static <T, R, C> CsvBulkWriter<T, R, C> forSchema(
                CsvMapper mapper,
                CsvSchema schema,
                Converter<T, R, C> converter,
                @Nullable C converterContext,
                FSDataOutputStream stream) {
            return new CsvBulkWriter<>(mapper, schema, converter, converterContext, stream);
        }

        static <T> CsvBulkWriter<T, T, Void> forPojo(Class<T> pojoClass, FSDataOutputStream stream) {
            final Converter<T, T, Void> converter = (value, Context) -> value;
            final CsvMapper csvMapper = JacksonMapperFactory.createCsvMapper();
            final CsvSchema schema = csvMapper.schemaFor(pojoClass).withoutQuoteChar();
            return new CsvBulkWriter<>(csvMapper, schema, converter, null, stream);
        }

        @Override
        public void addElement(T element) throws IOException {
            final R r = converter.convert(element, converterContext);
            generator.writeObject(r);
        }

        @Override
        public void flush() throws IOException {
            generator.flush();
        }

        @Override
        public void finish() throws IOException {
            generator.close();
            stream.sync();
        }
    }
}
