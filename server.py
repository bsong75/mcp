#!/usr/bin/env python3
"""
FastMCP Server with LangChain tools using langchain-mcp-adapters
"""

import math
import requests
from langchain_core.tools import tool
from langchain_mcp_adapters.tools import to_fastmcp
from mcp.server.fastmcp import FastMCP

# LangChain tool: Calculator
@tool
def calculate(expression: str) -> str:
    """
    Perform mathematical calculations and evaluate expressions.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., '2+2', 'sqrt(16)', 'sin(pi/2)')
    
    Returns:
        The result of the calculation
    """
    try:
        # Clean the expression
        expression = expression.replace('^', '**')  # Convert ^ to **
        expression = expression.replace(' ', '')     # Remove spaces
        
        if not expression:
            return "Error: Empty expression"
        
        # Security: Check for dangerous patterns
        dangerous_patterns = ['import', 'exec', 'eval', '__', 'open', 'file']
        for pattern in dangerous_patterns:
            if pattern in expression.lower():
                return f"Error: '{pattern}' not allowed in expressions"
        
        # Safe environment for evaluation
        safe_dict = {
            "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "asin": math.asin, "acos": math.acos, "atan": math.atan,
            "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
            "exp": math.exp, "pow": pow,
            "pi": math.pi, "e": math.e,
            "ceil": math.ceil, "floor": math.floor,
            "__builtins__": {}  # Security: disable built-ins
        }
        
        # Evaluate safely
        result = eval(expression, safe_dict)
        
        # Format result
        if isinstance(result, float):
            if result.is_integer():
                return f"{expression} = {int(result)}"
            else:
                return f"{expression} = {result:.10g}"
        else:
            return f"{expression} = {result}"
            
    except ZeroDivisionError:
        return "Error: Division by zero"
    except ValueError as e:
        return f"Error: Invalid value - {str(e)}"
    except SyntaxError:
        return "Error: Invalid mathematical expression"
    except Exception as e:
        return f"Error: {str(e)}"

# LangChain tool: Chat with Ollama
@tool
def chat_with_gemma3(message: str, model: str = "gemma3") -> str:
    """
    Chat with Gemma3 via Ollama.
    
    Args:
        message: The message to send to Gemma3
        model: The model to use (default: gemma3)
    
    Returns:
        Gemma3's response
    """
    url = "http://localhost:11434/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": message}],
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload, timeout=300)
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"

# LangChain tool: Web Search (optional - requires duckduckgo-search)
@tool 
def web_search(query: str) -> str:
    """
    Search the web for information.
    
    Args:
        query: Search query
        
    Returns:
        Search results summary
    """
    try:
        from duckduckgo_search import DDGS
        
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            
        if not results:
            return "No search results found."
        
        summary = f"Search results for '{query}':\n\n"
        for i, result in enumerate(results, 1):
            summary += f"{i}. {result['title']}\n"
            summary += f"   {result['body'][:200]}...\n"
            summary += f"   Source: {result['href']}\n\n"
        
        return summary
        
    except ImportError:
        return "Web search requires: pip install duckduckgo-search"
    except Exception as e:
        return f"Search error: {str(e)}"

# LangChain tool: Text summarizer
@tool
def summarize_text(text: str, max_length: int = 100) -> str:
    """
    Summarize a given text.
    
    Args:
        text: Text to summarize
        max_length: Maximum length of summary (default: 100 words)
        
    Returns:
        Summarized text
    """
    try:
        words = text.split()
        if len(words) <= max_length:
            return f"Text is already short ({len(words)} words): {text}"
        
        # Simple summarization: take first and last parts
        first_part = " ".join(words[:max_length//2])
        last_part = " ".join(words[-max_length//2:])
        
        summary = f"{first_part}... {last_part}"
        return f"Summary ({len(summary.split())} words): {summary}"
        
    except Exception as e:
        return f"Summarization error: {str(e)}"

# Convert LangChain tools to FastMCP tools
calculate_tool = to_fastmcp(calculate)
chat_tool = to_fastmcp(chat_with_gemma3)
search_tool = to_fastmcp(web_search)
summarize_tool = to_fastmcp(summarize_text)

# Create FastMCP server with LangChain tools
mcp = FastMCP(
    name="LangChain + Gemma3 MCP Server",
    tools=[calculate_tool, chat_tool, search_tool, summarize_tool]
)

# Run the server
if __name__ == "__main__":
    transport = "stdio"  # Change to "streamable-http" for HTTP transport
    
    if transport == "stdio":
        print("Running LangChain MCP server with stdio transport")
        mcp.run(transport="stdio")
    elif transport == "streamable-http":
        print("Running LangChain MCP server with streamable HTTP transport on port 8050")
        mcp.run(transport="streamable-http")
    else:
        raise ValueError(f"Unknown transport: {transport}")