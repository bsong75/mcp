#!/usr/bin/env python3
"""
Advanced Multi-Tab Gradio Frontend for LangChain MCP
"""

import gradio as gr
import asyncio
import sys
import threading
import queue
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

class AdvancedMCPInterface:
    def __init__(self):
        self.server_file = "server.py"  # Change to your server file
        
    async def call_tool_async(self, tool_name: str, arguments: dict):
        """Call MCP tool"""
        try:
            server_params = StdioServerParameters(
                command=sys.executable,
                args=[self.server_file]
            )
            
            async with stdio_client(server_params) as streams:
                read_stream, write_stream = streams
                
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments=arguments)
                    
                    if result.content and result.content[0].text:
                        return result.content[0].text
                    return "No response received"
                    
        except Exception as e:
            return f"Error: {str(e)}"
    
    def call_tool_sync(self, tool_name: str, arguments: dict):
        """Sync wrapper"""
        def run_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self.call_tool_async(tool_name, arguments)
                )
            finally:
                loop.close()
        
        result_queue = queue.Queue()
        
        def thread_func():
            try:
                result = run_async()
                result_queue.put(("success", result))
            except Exception as e:
                result_queue.put(("error", str(e)))
        
        thread = threading.Thread(target=thread_func)
        thread.start()
        thread.join(timeout=300)
        
        if thread.is_alive():
            return "⏱️ Timeout: Request took too long"
        
        try:
            status, result = result_queue.get_nowait()
            return result if status == "success" else f"❌ {result}"
        except queue.Empty:
            return "❌ No response received"

# Global interface
mcp = AdvancedMCPInterface()

# Tab 1: Chat Interface
def chat_response(message, history):
    """Handle chat messages"""
    if not message.strip():
        return history, ""
    
    result = mcp.call_tool_sync("chat_with_gemma3", {"message": message})
    history.append((message, result))
    return history, ""

# Tab 2: Calculator
def calculate(expression):
    """Handle calculator input"""
    if not expression.strip():
        return "Please enter a mathematical expression"
    
    return mcp.call_tool_sync("calculate", {"expression": expression})

# Tab 3: Web Search
def web_search(query):
    """Handle web search"""
    if not query.strip():
        return "Please enter a search query"
    
    return mcp.call_tool_sync("web_search", {"query": query})

# Tab 4: Text Summarization
def summarize(text):
    """Handle text summarization"""
    if not text.strip():
        return "Please enter text to summarize"
    
    return mcp.call_tool_sync("summarize_text", {"text": text})

def create_advanced_interface():
    """Create advanced multi-tab interface"""
    
    with gr.Blocks(title="Advanced LangChain MCP", theme=gr.themes.Soft()) as interface:
        
        gr.Markdown(
            """
            # 🚀 Advanced LangChain MCP Interface
            
            **Multi-tool AI assistant powered by LangChain + MCP + Gemma3**
            """
        )
        
        with gr.Tabs():
            
            # Tab 1: Chat
            with gr.TabItem("💬 Chat with Gemma3"):
                gr.Markdown("### Direct conversation with Gemma3")
                
                chatbot = gr.Chatbot(
                    height=400,
                    avatar_images=("👤", "🤖")
                )
                
                with gr.Row():
                    chat_input = gr.Textbox(
                        placeholder="Ask anything...",
                        scale=4,
                        show_label=False
                    )
                    chat_send = gr.Button("Send", variant="primary", scale=1)
                
                gr.Button("Clear Chat").click(lambda: [], outputs=chatbot)
                
                # Chat examples
                gr.Examples(
                    examples=[
                        "What is machine learning?",
                        "Explain quantum computing",
                        "Write a Python function to sort a list",
                        "What are the benefits of renewable energy?"
                    ],
                    inputs=chat_input
                )
                
                chat_send.click(chat_response, [chat_input, chatbot], [chatbot, chat_input])
                chat_input.submit(chat_response, [chat_input, chatbot], [chatbot, chat_input])
            
            # Tab 2: Calculator
            with gr.TabItem("🧮 Calculator"):
                gr.Markdown("### Advanced Mathematical Calculator")
                
                calc_input = gr.Textbox(
                    label="Mathematical Expression",
                    placeholder="e.g., sqrt(16) + 2^3, sin(pi/2), log(100)",
                    lines=2
                )
                
                calc_button = gr.Button("Calculate", variant="primary")
                calc_output = gr.Textbox(label="Result", lines=3)
                
                # Calculator examples
                gr.Examples(
                    examples=[
                        "2 + 2 * 3",
                        "sqrt(16) + 2^3", 
                        "sin(pi/2)",
                        "log(100)",
                        "(5 + 3) * (10 - 2)"
                    ],
                    inputs=calc_input
                )
                
                calc_button.click(calculate, calc_input, calc_output)
                calc_input.submit(calculate, calc_input, calc_output)
            
            # Tab 3: Web Search
            with gr.TabItem("🔍 Web Search"):
                gr.Markdown("### Search the web for current information")
                
                search_input = gr.Textbox(
                    label="Search Query",
                    placeholder="e.g., latest AI news, Python tutorials, weather forecast",
                    lines=2
                )
                
                search_button = gr.Button("Search", variant="primary")
                search_output = gr.Textbox(label="Search Results", lines=10)
                
                # Search examples
                gr.Examples(
                    examples=[
                        "latest AI breakthroughs 2024",
                        "Python programming tutorials",
                        "climate change news",
                        "best programming languages 2024"
                    ],
                    inputs=search_input
                )
                
                search_button.click(web_search, search_input, search_output)
                search_input.submit(web_search, search_input, search_output)
            
            # Tab 4: Text Summarization
            with gr.TabItem("📝 Summarizer"):
                gr.Markdown("### Summarize long texts into key points")
                
                with gr.Row():
                    with gr.Column():
                        text_input = gr.Textbox(
                            label="Text to Summarize",
                            placeholder="Paste your long text here...",
                            lines=10
                        )
                        
                        summarize_button = gr.Button("Summarize", variant="primary")
                    
                    with gr.Column():
                        summary_output = gr.Textbox(
                            label="Summary",
                            lines=10
                        )
                
                summarize_button.click(summarize, text_input, summary_output)
            
            # Tab 5: Multi-Tool
            with gr.TabItem("🛠️ Multi-Tool"):
                gr.Markdown("### Use multiple tools in one interface")
                
                multi_chatbot = gr.Chatbot(
                    height=400,
                    avatar_images=("👤", "🔧")
                )
                
                multi_input = gr.Textbox(
                    placeholder="Type naturally or use commands: calc:, search:, summarize:",
                    scale=4,
                    show_label=False
                )
                
                multi_send = gr.Button("Send", variant="primary")
                
                def multi_tool_response(message, history):
                    """Handle multi-tool messages"""
                    if not message.strip():
                        return history, ""
                    
                    # Route to appropriate tool
                    if message.startswith('calc:'):
                        result = calculate(message[5:].strip())
                        history.append((message, f"🧮 {result}"))
                    elif message.startswith('search:'):
                        result = web_search(message[7:].strip())
                        history.append((message, f"🔍 {result}"))
                    elif message.startswith('summarize:'):
                        result = summarize(message[10:].strip())
                        history.append((message, f"📝 {result}"))
                    else:
                        result = mcp.call_tool_sync("chat_with_gemma3", {"message": message})
                        history.append((message, f"🤖 {result}"))
                    
                    return history, ""
                
                multi_send.click(multi_tool_response, [multi_input, multi_chatbot], [multi_chatbot, multi_input])
                multi_input.submit(multi_tool_response, [multi_input, multi_chatbot], [multi_chatbot, multi_input])
                
                gr.Markdown(
                    """
                    **Commands:**
                    - `calc: 2+2` for calculator
                    - `search: AI news` for web search  
                    - `summarize: [text]` for summarization
                    - Regular text for chat
                    """
                )
    
    return interface

def main():
    """Launch the advanced interface"""
    print("🚀 Starting Advanced LangChain MCP Interface...")
    
    interface = create_advanced_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )

if __name__ == "__main__":
    main()