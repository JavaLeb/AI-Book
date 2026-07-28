package flink.application.study.table.sqlgateway;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.Statement;
public class HiveServer2EndpointConnection {
    public static void main(String[] args) throws Exception {
        try (
                // Please replace the JDBC URI with your actual host, port and database.
                Connection connection = DriverManager.getConnection("jdbc:hive2://slave-2:10000/default;auth=none");
                Statement statement = connection.createStatement()) {
            statement.execute("SHOW DATABASES");
            ResultSet resultSet = statement.getResultSet();
            while (resultSet.next()) {
                System.out.println(resultSet.getString(1));
            }
        }
    }
}
