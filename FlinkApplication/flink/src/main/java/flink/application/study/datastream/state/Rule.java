package flink.application.study.datastream.state;

public class Rule {
    public String name;

    public  Shape first;
    public  Shape second;


    public Rule(String name, Shape first, Shape second) {
        this.name = name;
        this.first = first;
        this.second = second;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

}
