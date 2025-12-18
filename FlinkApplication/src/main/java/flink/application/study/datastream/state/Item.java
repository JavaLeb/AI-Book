package flink.application.study.datastream.state;

public class Item {
    private Color color;
    private Shape shape;

    public Item(Color color, Shape shape) {
        this.color = color;
        this.shape = shape;
    }

    public Color getColor() {
        return color;
    }

    public void setColor(Color color) {
        this.color = color;
    }

    public Shape getShape() {
        return shape;
    }

    public void setShape(Shape shape) {
        this.shape = shape;
    }

    @Override
    public String toString() {
        return "Item{" +
                "color=" + color.getName() +
                ", shape=" + shape.getName() +
                '}';
    }

}
