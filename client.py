#!/usr/bin/env python3
import asyncio
import sys
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

async def main():
    """Main client function"""
    try:
        # Create server parameters
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["server.py"]
        )
        
        # Connect to the server using stdio
        async with stdio_client(server_params) as streams:
            
            read_stream, write_stream = streams
            
            async with ClientSession(read_stream, write_stream) as session:
                # Initialize the connection
                print("🔗 Connecting to LangChain MCP server...")
                await session.initialize()
                print("✅ Connected successfully!")
                
                # List available tools
                tools_result = await session.list_tools()
                print("\n🔧 Available LangChain tools:")
                for tool in tools_result.tools:
                    print(f"  - {tool.name}: {tool.description}")
                
                print("\n" + "="*70)
                print("   LangChain + MCP Chat with Gemma3")
                print("   Commands:")
                print("   - Type normally to chat with Gemma3")
                print("   - 'calc: <expression>' for calculator")
                print("   - 'search: <query>' for web search")
                print("   - 'summarize: <text>' for text summarization")
                print("   - 'tools' to list all tools")
                print("   - 'quit' to exit")
                print("="*70)
                
                # Chat loop
                while True:
                    try:
                        user_input = input("\nYou: ").strip()
                        
                        if user_input.lower() in ['quit', 'exit', 'q']:
                            print("Goodbye!")
                            break
                        
                        if not user_input:
                            continue
                        
                        if user_input.lower() == 'tools':
                            print("🔧 Available LangChain tools:")
                            for tool in tools_result.tools:
                                print(f"  - {tool.name}: {tool.description}")
                            continue
                        
                        # Calculator command
                        if user_input.lower().startswith('calc:'):
                            expression = user_input[5:].strip()
                            if expression:
                                print("🧮 Calculator: ", end="", flush=True)
                                result = await session.call_tool("calculate", arguments={"expression": expression})
                                print(result.content[0].text)
                            else:
                                print("Please provide an expression. Example: calc: 2+2")
                            continue
                        
                        # Web search command
                        if user_input.lower().startswith('search:'):
                            query = user_input[7:].strip()
                            if query:
                                print("🔍 Searching: ", end="", flush=True)
                                result = await session.call_tool("web_search", arguments={"query": query})
                                print(result.content[0].text)
                            else:
                                print("Please provide a search query. Example: search: python programming")
                            continue
                        
                        # Summarize command
                        if user_input.lower().startswith('summarize:'):
                            text = user_input[10:].strip()
                            if text:
                                print("📝 Summarizing: ", end="", flush=True)
                                result = await session.call_tool("summarize_text", arguments={"text": text})
                                print(result.content[0].text)
                            else:
                                print("Please provide text to summarize. Example: summarize: <your text here>")
                            continue
                        
                        # Regular chat with Gemma3
                        print("🤖 Gemma3: ", end="", flush=True)
                        result = await session.call_tool("chat_with_gemma3", arguments={"message": user_input})
                        print(result.content[0].text)
                        
                    except KeyboardInterrupt:
                        print("\n\nGoodbye!")
                        break
                    except EOFError:
                        print("\nGoodbye!")
                        break
                        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())