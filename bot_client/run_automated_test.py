#!/usr/bin/env python3
"""
Automated Test Suite for Bot API - runs bot_api_test.py without user interaction

This is a wrapper around the interactive bot_api_test.py that:
1. Uses all models from .env
2. Uses all MCP servers from .env  
3. Uses prompts from test_prompts.txt
4. Runs all tests automatically
5. Shows results summary

Usage:
    python bot_client/run_automated_test.py
"""

import asyncio
import sys
import os
from collections import defaultdict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text

# Add parent directory to import bot modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot_client.bot_api_test import BotAPITestTool

console = Console()

def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate estimated cost for a model based on token usage"""
    # Pricing per 1M tokens - matches _estimate_cost in bot_api_test.py
    pricing = {
        'gpt-4o-mini': {'prompt': 0.150, 'completion': 0.600},
        'gpt-4o': {'prompt': 2.50, 'completion': 10.00},
        'gpt-5-mini': {'prompt': 0.150, 'completion': 0.600},  # Assuming similar to gpt-4o-mini
        'gpt-5-nano': {'prompt': 0.050, 'completion': 0.200},   # Assuming cheaper
        'gemini/gemini-2.5-flash-lite': {'prompt': 0.050, 'completion': 0.200},
        'gemini/gemini-2.5-flash': {'prompt': 0.100, 'completion': 0.400},
        'xai/grok-3-mini': {'prompt': 0.500, 'completion': 1.500},
        'xai/grok-3-mini-fast': {'prompt': 0.300, 'completion': 0.900},
    }
    
    # Default pricing if model not found
    default_pricing = {'prompt': 0.150, 'completion': 0.600}
    
    # Try to find matching pricing (partial match like _estimate_cost)
    model_lower = model.lower()
    matched_pricing = None
    for model_key, prices in pricing.items():
        if model_key in model_lower:
            matched_pricing = prices
            break
    
    model_pricing = matched_pricing or default_pricing
    
    prompt_cost = (prompt_tokens / 1_000_000) * model_pricing['prompt']
    completion_cost = (completion_tokens / 1_000_000) * model_pricing['completion']
    
    return prompt_cost + completion_cost

def create_dashboard_report(results, models, prompts, mcp_servers):
    """Create a comprehensive dashboard report"""
    
    # Calculate overall statistics
    total_tests = len(results)
    successful_tests = len([r for r in results if r['success']])
    failed_tests = total_tests - successful_tests
    success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
    
    # Group results by model and prompt
    results_by_model = defaultdict(list)
    results_by_prompt = defaultdict(list)
    
    for result in results:
        results_by_model[result['model']].append(result)
        results_by_prompt[result['prompt']].append(result)
    
    # Analyze function calls and structured output
    function_call_analysis = analyze_function_calls(results)
    structured_output_analysis = analyze_structured_output(results)
    
    # Create dashboard
    console.print("\n" + "="*100)
    console.print("🎯 AUTOMATED TEST DASHBOARD")
    console.print("="*100)
    
    # Calculate overall token and cost statistics
    successful_results = [r for r in results if r['success']]
    total_tokens_used = sum(r.get('tokens_used', 0) for r in successful_results)
    total_prompt_tokens = sum(r.get('prompt_tokens', 0) for r in successful_results)
    total_completion_tokens = sum(r.get('completion_tokens', 0) for r in successful_results)
    total_estimated_cost = sum(
        calculate_cost(r['model'], r.get('prompt_tokens', 0), r.get('completion_tokens', 0))
        for r in successful_results
    )
    
    # Overall Summary
    summary_panel = Panel.fit(
        f"[bold cyan]Overall Summary[/bold cyan]\n\n"
        f"📊 Total Tests: {total_tests}\n"
        f"✅ Successful: {successful_tests}\n"
        f"❌ Failed: {failed_tests}\n"
        f"📈 Success Rate: {success_rate:.1f}%\n\n"
        f"🤖 Models Tested: {len(models)}\n"
        f"💬 Prompts Used: {len(prompts)}\n"
        f"🔧 MCP Servers: {len(mcp_servers)}\n\n"
        f"[dim]Token Usage (Successful Tests):[/dim]\n"
        f"🔢 Total Tokens: {total_tokens_used:,}\n"
        f"📥 Prompt Tokens: {total_prompt_tokens:,}\n"
        f"📤 Completion Tokens: {total_completion_tokens:,}\n"
        f"💰 Est. Total Cost: ${total_estimated_cost:.4f}",
        title="📈 Test Overview",
        border_style="cyan"
    )
    console.print(summary_panel)
    
    # Model Performance Table
    model_table = Table(title="🏆 Model Performance", show_header=True, header_style="bold magenta")
    model_table.add_column("Model", style="cyan", width=20, no_wrap=True)
    model_table.add_column("Tests", justify="right", style="white", width=6)
    model_table.add_column("Success", justify="right", style="green", width=7)
    model_table.add_column("Fail", justify="right", style="red", width=5)
    model_table.add_column("Rate", justify="right", style="yellow", width=6)
    model_table.add_column("Avg Time", justify="right", style="blue", width=8)
    model_table.add_column("Tokens", justify="right", style="magenta", width=8)
    model_table.add_column("Prompt", justify="right", style="cyan", width=7)
    model_table.add_column("Comp", justify="right", style="green", width=6)
    model_table.add_column("Cached", justify="right", style="blue", width=6)
    model_table.add_column("Cost", justify="right", style="yellow", width=12)
    
    for model in sorted(models):
        model_results = results_by_model[model]
        success_count = len([r for r in model_results if r['success']])
        fail_count = len(model_results) - success_count
        rate = (success_count / len(model_results) * 100) if model_results else 0
        avg_time = sum(r['duration'] for r in model_results) / len(model_results) if model_results else 0
        
        # Calculate token statistics for successful tests only
        successful_results = [r for r in model_results if r['success']]
        if successful_results:
            total_tokens = sum(r.get('tokens_used') or 0 for r in successful_results)
            total_prompt_tokens = sum(r.get('prompt_tokens') or 0 for r in successful_results)
            total_completion_tokens = sum(r.get('completion_tokens') or 0 for r in successful_results)
            total_cached_tokens = sum(r.get('cached_tokens') or 0 for r in successful_results)
            
            # Calculate total estimated cost
            total_cost = sum(
                calculate_cost(model, r.get('prompt_tokens', 0), r.get('completion_tokens', 0))
                for r in successful_results
            )
        else:
            total_tokens = 0
            total_prompt_tokens = 0
            total_completion_tokens = 0
            total_cached_tokens = 0
            total_cost = 0
        
        model_table.add_row(
            model[:19],  # Truncate model name if too long
            str(len(model_results)),
            str(success_count),
            str(fail_count),
            f"{rate:.1f}%",
            f"{avg_time:.2f}s",
            str(total_tokens),
            str(total_prompt_tokens),
            str(total_completion_tokens),
            str(total_cached_tokens),
            f"${total_cost:.6f}"
        )
    
    console.print("\n")
    console.print(model_table)
    
    # Prompt Performance Table
    if len(prompts) > 1:
        prompt_table = Table(title="💬 Prompt Performance", show_header=True, header_style="bold magenta")
        prompt_table.add_column("Prompt", style="cyan", width=40)
        prompt_table.add_column("Tests", justify="right", style="white")
        prompt_table.add_column("Success", justify="right", style="green")
        prompt_table.add_column("Fail", justify="right", style="red")
        prompt_table.add_column("Rate", justify="right", style="yellow")
        
        for prompt in sorted(prompts):
            prompt_results = results_by_prompt[prompt]
            success_count = len([r for r in prompt_results if r['success']])
            fail_count = len(prompt_results) - success_count
            rate = (success_count / len(prompt_results) * 100) if prompt_results else 0
            
            # Truncate long prompts for display
            display_prompt = prompt[:37] + "..." if len(prompt) > 37 else prompt
            
            prompt_table.add_row(
                display_prompt,
                str(len(prompt_results)),
                str(success_count),
                str(fail_count),
                f"{rate:.1f}%"
            )
        
        console.print("\n")
        console.print(prompt_table)
    
    # Function Call Analysis
    if function_call_analysis['total_with_tools'] > 0:
        func_panel = Panel.fit(
            f"[bold cyan]Function Call Analysis[/bold cyan]\n\n"
            f"🔧 Tests with Tools: {function_call_analysis['total_with_tools']}\n"
            f"✅ Tools Called: {function_call_analysis['tools_called']}\n"
            f"❌ Tools Available but Not Used: {function_call_analysis['tools_available_not_used']}\n"
            f"📊 Tool Call Rate: {function_call_analysis['tool_call_rate']:.1f}%",
            title="🔧 Function Calls",
            border_style="green"
        )
        console.print("\n")
        console.print(func_panel)
    
    # Structured Output Analysis
    struct_panel = Panel.fit(
        f"[bold cyan]Structured Output Analysis[/bold cyan]\n\n"
        f"📝 Total Responses: {structured_output_analysis['total_responses']}\n"
        f"✅ Valid Structure: {structured_output_analysis['valid_structure']}\n"
        f"❌ Invalid Structure: {structured_output_analysis['invalid_structure']}\n"
        f"📊 Structure Rate: {structured_output_analysis['structure_rate']:.1f}%\n\n"
        f"[dim]Response Types:[/dim]\n{structured_output_analysis['type_breakdown']}",
        title="📝 Structured Output",
        border_style="blue"
    )
    console.print("\n")
    console.print(struct_panel)
    
    # Detailed Failures
    failures = [r for r in results if not r['success']]
    if failures:
        console.print("\n[red]❌ Detailed Failures:[/red]")
        
        for i, failure in enumerate(failures[:10], 1):  # Show first 10 failures
            error_msg = failure.get('error', 'Unknown error')
            error_msg = error_msg[:100] + "..." if len(error_msg) > 100 else error_msg
            
            failure_panel = Panel.fit(
                f"[bold]Model:[/bold] {failure['model']}\n"
                f"[bold]Prompt:[/bold] {failure['prompt'][:50]}{'...' if len(failure['prompt']) > 50 else ''}\n"
                f"[bold]Duration:[/bold] {failure['duration']:.2f}s\n"
                f"[bold]Error:[/bold] {error_msg}",
                title=f"❌ Failure #{i}",
                border_style="red"
            )
            console.print(failure_panel)
        
        if len(failures) > 10:
            console.print(f"[dim]... and {len(failures) - 10} more failures[/dim]")

def analyze_function_calls(results):
    """Analyze function call performance"""
    total_with_tools = 0
    tools_called = 0
    tools_available_not_used = 0
    
    for result in results:
        llm_calls = result.get('llm_calls', [])
        for call in llm_calls:
            if call.get('tools_available', False):
                total_with_tools += 1
                if call.get('tool_calls'):
                    tools_called += 1
                else:
                    tools_available_not_used += 1
    
    tool_call_rate = (tools_called / total_with_tools * 100) if total_with_tools > 0 else 0
    
    return {
        'total_with_tools': total_with_tools,
        'tools_called': tools_called,
        'tools_available_not_used': tools_available_not_used,
        'tool_call_rate': tool_call_rate
    }

def analyze_structured_output(results):
    """Analyze structured output performance"""
    total_responses = len([r for r in results if r['success']])
    valid_structure = 0
    invalid_structure = 0
    response_types = defaultdict(int)
    
    for result in results:
        if result['success']:
            response_type = result.get('response_type')
            if response_type:
                valid_structure += 1
                response_types[response_type] += 1
            else:
                invalid_structure += 1
    
    structure_rate = (valid_structure / total_responses * 100) if total_responses > 0 else 0
    
    # Format type breakdown
    type_breakdown = ""
    for resp_type, count in sorted(response_types.items()):
        type_breakdown += f"• {resp_type}: {count}\n"
    type_breakdown = type_breakdown.rstrip()
    
    return {
        'total_responses': total_responses,
        'valid_structure': valid_structure,
        'invalid_structure': invalid_structure,
        'structure_rate': structure_rate,
        'type_breakdown': type_breakdown
    }

async def main():
    """Run automated tests using BotAPITestTool"""
    try:
        # Create the test tool
        tool = BotAPITestTool()
        
        # Get configuration
        models = tool.config['default_models']
        mcp_servers = tool.config['mcp_servers']
        prompts = tool.load_prompts()
        
        console.print(f"\n[bold cyan]🤖 Automated Test Suite[/bold cyan]")
        console.print(f"[dim]Models: {len(models)} | Prompts: {len(prompts)} | MCP Servers: {len(mcp_servers)}[/dim]\n")
        
        # Run tests with default settings (show_raw=False, show_stats=False, show_tool_details=False)
        results = await tool.execute_tests(models, prompts, mcp_servers, False, False, False)
        
        # Create comprehensive dashboard report
        create_dashboard_report(results, models, prompts, mcp_servers)
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Tests interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ Error: {e}[/red]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
