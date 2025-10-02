from fastmcp import FastMCP

mcp = FastMCP("My MCP Server")

@mcp.tool()
def greet(name: str) -> str:
    """Greet a user by name."""
    return f"Hello, {name}!"

@mcp.tool()
def get_weather(location: str) -> str:
    """Get the current weather description for a given location."""
    # Simulate getting weather
    return f"The weather in {location} is sunny and 25°C."

@mcp.tool()
def tell_joke() -> str:
    """Tell a light-hearted, family-friendly joke."""
    return "Why don't scientists trust atoms? Because they make up everything!"

@mcp.tool()
def transfer_call() -> str:
    """Transfer the caller to a human agent."""
    return "Transferring your call to a human agent now."

if __name__ == "__main__":
    import uvicorn
    app = mcp.sse_app
    uvicorn.run(app, host="127.0.0.1", port=8000)
