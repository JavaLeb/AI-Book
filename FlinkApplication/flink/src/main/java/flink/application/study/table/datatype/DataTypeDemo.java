package flink.application.study.table.datatype;

import org.apache.flink.table.types.DataType;

import static org.apache.flink.table.api.DataTypes.*;

public class DataTypeDemo {
    public static void main(String[] args) {
        DataType t = INTERVAL(DAY(), SECOND(3));
        DataType charDT = CHAR(10);
    }
}
