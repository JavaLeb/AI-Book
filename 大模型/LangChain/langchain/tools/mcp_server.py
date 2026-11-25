from fastmcp import FastMCP

mcp = FastMCP("My Tools Server")

@mcp.tool()
def add(a: float, b: float) -> float:
    """执行浮点数加法运算"""
    return a + b

@mcp.tool()
def get_weather(city: str) -> dict:
    """查询指定城市的天气信息"""
    # 这里可以接入真实 API、数据库、内部系统等
    return {"city": city, "temperature": 25, "condition": "sunny"}

if __name__ == "__main__":
    mcp.run()   # 默认 stdio 传输