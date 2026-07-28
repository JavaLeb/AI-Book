package flink.application.study.table;

import org.apache.flink.table.types.DataType;

import static org.apache.flink.table.api.DataTypes.*;

public class TableAPIDataTypeDemo {
    public static void main(String[] args) {
        DataType t = INTERVAL(DAY(), SECOND(3));
    }
}
