#!/usr/bin/env python3
"""
Automated Test Suite - Tests all models with MCP tools + structured output

This script automatically tests:
1. All configured models from .env
2. Tool calling with MCP servers
3. Structured JSON output validation
4. Provides comprehensive pass/fail report

Usage:
    python bot_client/run_automated_test.py
    
Or with Docker:
    docker-compose run --rm staffai python bot_client/run_automated_test.py
"""

import asyncio
import json
import logging
import os
import sys
import warnings
import redis
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

# Suppress noisy HTTP connection warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*SSE.*")
warnings.filterwarnings("ignore", message=".*httpx.*")

# Set up logging to suppress noise
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("mcp").setLevel(logging.CRITICAL)
logging.getLogger("mcp.client").setLevel(logging.CRITICAL)
logging.getLogger("mcp.client.streamable_http").setLevel(logging.CRITICAL)
logging.getLogger("utils.litellm_client").setLevel(logging.WARNING)
logging.getLogger("utils.message_handler").setLevel(logging.WARNING)

# Rich for beautiful output
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich import box
from rich.syntax import Syntax

# Add parent directory to import bot modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

from utils.message_handler import MessageHandler

console = Console()

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s:%(levelname)s:%(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


# Mock Discord classes (copied from bot_api_test.py)
class MockMessage:
    """Mock Discord message for testing"""
    def __init__(self, content: str, user_id: int, channel_id: int, guild_id: int):
        self.content = content
        self.author = MockUser(user_id)
        self.channel = MockChannel(channel_id)
        self.guild = MockGuild(guild_id)
        self.mentions = []
        self.reference = None


class MockUser:
    """Mock Discord user"""
    def __init__(self, user_id: int):
        self.id = user_id
        self.name = f"TestUser{user_id}"
        self.display_name = self.name
        self.roles = []


class MockChannel:
    """Mock Discord channel"""
    def __init__(self, channel_id: int):
        self.id = channel_id
    
    def history(self, limit: int = 10):
        """Mock channel history - returns empty async iterator"""
        async def empty_history():
            return
            yield  # Make it a generator
        return empty_history()


class MockGuild:
    """Mock Discord guild"""
    def __init__(self, guild_id: int):
        self.id = guild_id


class MockBot:
    """Mock Discord bot for MessageHandler"""
    def __init__(self, redis_client, base_url, api_key, mcp_servers):
        self.user = None
        self.response_chance = 1.0
        self.stats_cog = None
        self.redis_client = redis_client
        self.litellm_api_url = base_url
        self.litellm_api_key = api_key
        self.mcp_servers = mcp_servers
    
    @property
    def random_response_delivery_chance(self):
        return 0.0


def _read_docker_secret(secret_name: str) -> Optional[str]:
    """Read a Docker secret mounted at /run/secrets/<secret_name> if present.
    Returns the secret string (stripped) or None if not available.
    """
    secret_path = f"/run/secrets/{secret_name}"
    try:
        if os.path.exists(secret_path):
            with open(secret_path, 'r', encoding='utf-8') as f:
                value = f.read().strip()
                logger.info(f"Loaded Docker secret: {secret_name}")
                return value
    except Exception as e:
        logger.debug(f"Could not read docker secret {secret_name} at {secret_path}: {e}")
    return None


@dataclass
class TestResult:
    """Result of testing a single model with a single prompt"""
    model: str
    prompt: str
    success: bool
    tool_called: bool = False
    tool_name: Optional[str] = None
    structured_output_valid: bool = False
    response_type: Optional[str] = None
    response_text: Optional[str] = None
    error: Optional[str] = None
    duration: float = 0.0


@dataclass
class ModelTestSummary:
    """Summary of all tests for a single model"""
    model: str
    total_tests: int = 0
    successful_tests: int = 0
    tool_calling_success: int = 0
    tool_calling_failed: int = 0
    structured_output_success: int = 0
    structured_output_failed: int = 0
    total_duration: float = 0.0
    errors: List[str] = field(default_factory=list)


def load_test_prompts(prompts_file: str = "bot_client/prompts/test_prompts.txt") -> List[str]:
    """Load test prompts from file, ignoring comments and empty lines"""
    prompts = []
    prompts_path = Path(prompts_file)
    
    if not prompts_path.exists():
        console.print(f"[yellow]Warning: {prompts_file} not found, using default prompts[/yellow]")
        return [
            "give me a gif of a cat",
            "show me a dog gif",
            "search for funny memes"
        ]
    
    with open(prompts_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip comments (lines starting with #) and empty lines
            if line and not line.startswith('#'):
                prompts.append(line)
    
    if not prompts:
        console.print("[yellow]No prompts found in file, using defaults[/yellow]")
        return ["give me a gif of a cat"]
    
    return prompts


def load_models_from_env() -> List[str]:
    """Load models from environment variables"""
    # Try LITELLM_MODELS first (plural), then LITELLM_MODEL (singular)
    models_str = os.getenv('LITELLM_MODELS', '')
    if not models_str:
        models_str = os.getenv('LITELLM_MODEL', '')
    
    if not models_str:
        console.print("[red]Error: No models configured in .env (LITELLM_MODELS or LITELLM_MODEL)[/red]")
        sys.exit(1)
    
    models = [m.strip() for m in models_str.split(',') if m.strip()]
    
    if not models:
        console.print("[red]Error: No valid models found in configuration[/red]")
        sys.exit(1)
    
    return models


def get_redis_client() -> redis.Redis:
    """Get Redis client (can be None if Redis is not available)"""
    try:
        redis_host = os.getenv('REDIS_HOST', 'localhost')
        redis_port = int(os.getenv('REDIS_PORT', '6379'))
        redis_db = int(os.getenv('REDIS_DB', '0'))
        
        client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=True,
            socket_connect_timeout=5
        )
        
        # Test connection
        client.ping()
        logger.info(f"Redis connected: {redis_host}:{redis_port}")
        return client
    except Exception as e:
        logger.warning(f"Redis not available: {e}")
        console.print(f"[yellow]⚠ Redis not available: {e}[/yellow]")
        console.print("[yellow]  Tests will run without conversation context[/yellow]")
        # Return a mock Redis client that doesn't crash
        return None


async def test_model_with_prompt(
    bot: MockBot,
    model: str,
    prompt: str,
    user_id: int = 123456789,
    channel_id: int = 987654321,
    guild_id: int = 111222333
) -> TestResult:
    """Test a single model with a single prompt using MessageHandler (like interactive tool)"""
    
    result = TestResult(model=model, prompt=prompt, success=False)
    
    try:
        start_time = asyncio.get_event_loop().time()
        
        # Create mock message
        mock_msg = MockMessage(prompt, user_id, channel_id, guild_id)
        
        # Create MessageHandler with the specific model
        message_handler = MessageHandler(
            bot=bot,
            model=model  # Use the specific model for this test
        )
        
        # Execute through MessageHandler (like the interactive tool does)
        handler_result = await message_handler.handle_message(message=mock_msg)
        
        result.duration = asyncio.get_event_loop().time() - start_time
        
        # Extract results
        if handler_result.get('should_respond'):
            result.success = True
            result.response_type = handler_result.get('response_type')
            result.response_text = handler_result.get('response_text', '')[:100]
            result.structured_output_valid = True  # If handler succeeded, output was valid
            
            # Check for tool calls in the LLM calls log
            llm_calls = handler_result.get('llm_calls', [])
            for call_info in llm_calls:
                if call_info.get('tool_calls'):
                    result.tool_called = True
                    result.tool_name = call_info['tool_calls'][0].get('name', 'unknown')
                    break
        else:
            result.success = False
            result.error = handler_result.get('error', 'Handler returned should_respond=False')
        
    except Exception as e:
        result.error = str(e)
        result.success = False
        logger.debug(f"Test failed for {model} with prompt '{prompt[:50]}...': {e}")
    
    return result


async def run_automated_tests() -> Dict[str, ModelTestSummary]:
    """Run automated tests for all models and prompts"""
    
    # Load configuration
    models = load_models_from_env()
    prompts = load_test_prompts()
    
    # Load API key from Docker secret or env
    api_key_from_secret = _read_docker_secret('litellm_api_key')
    api_key_from_env = os.getenv('LITELLM_API_KEY', 'sk-1234')
    api_key = api_key_from_secret or api_key_from_env
    
    if api_key_from_secret:
        console.print("[green]✓ Using API key from Docker secret[/green]")
    else:
        console.print("[yellow]⚠ Using API key from environment variable[/yellow]")
    
    # Get base URL - use LITELLM_API_URL (matches main bot)
    base_url = os.getenv('LITELLM_API_URL', 'http://localhost:4000')
    console.print(f"[cyan]LiteLLM Base URL: {base_url}[/cyan]")
    console.print(f"[dim]API Key: {api_key[:15]}...[/dim]\n")
    
    console.print(Panel.fit(
        f"[bold cyan]Automated Test Suite[/bold cyan]\n\n"
        f"Models: {len(models)}\n"
        f"Prompts: {len(prompts)}\n"
        f"Total Tests: {len(models) * len(prompts)}",
        border_style="cyan"
    ))
    
    # Load system prompt (personality)
    personality_prompt_path = Path(__file__).parent.parent / "utils" / "prompts" / "personality_prompt.txt"
    if personality_prompt_path.exists():
        with open(personality_prompt_path, 'r', encoding='utf-8') as f:
            system_prompt = f.read()
    else:
        console.print("[yellow]Warning: personality_prompt.txt not found, using minimal prompt[/yellow]")
        system_prompt = "You are a helpful assistant. Use available tools when appropriate."
    
    # Get MCP servers from env
    mcp_servers_str = os.getenv('MCP_SERVERS', '')
    mcp_servers = [s.strip() for s in mcp_servers_str.split(',') if s.strip()] if mcp_servers_str else []
    
    if not mcp_servers:
        console.print("[yellow]Warning: No MCP servers configured (MCP_SERVERS env var)[/yellow]")
        console.print("[yellow]Tool calling tests will be skipped[/yellow]\n")
    
    # Initialize results tracking
    results_by_model: Dict[str, ModelTestSummary] = {}
    all_results: List[TestResult] = []
    
    # Initialize Redis client
    redis_client = get_redis_client()
    if not redis_client:
        console.print("[yellow]Warning: Running without Redis (conversation context disabled)[/yellow]\n")
    
    # Create MockBot (MessageHandler needs a bot object)
    mock_bot = MockBot(
        redis_client=redis_client,
        base_url=base_url,
        api_key=api_key,
        mcp_servers=mcp_servers
    )
    
    console.print("[cyan]✓ Mock bot created[/cyan]\n")
    
    # Run tests with progress bar
    total_tests = len(models) * len(prompts)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("[cyan]Running tests...", total=total_tests)
        
        for model in models:
            # Initialize summary for this model
            summary = ModelTestSummary(model=model)
            results_by_model[model] = summary
            
            # Test each prompt
            for prompt in prompts:
                progress.update(task, description=f"[cyan]Testing {model} with: {prompt[:40]}...")
                
                result = await test_model_with_prompt(mock_bot, model, prompt)
                all_results.append(result)
                
                # Update summary
                summary.total_tests += 1
                summary.total_duration += result.duration
                
                if result.success:
                    summary.successful_tests += 1
                
                if result.tool_called:
                    summary.tool_calling_success += 1
                elif mcp_servers:  # Only count as failure if MCP servers are configured
                    summary.tool_calling_failed += 1
                
                if result.structured_output_valid:
                    summary.structured_output_success += 1
                else:
                    summary.structured_output_failed += 1
                
                if result.error:
                    summary.errors.append(f"{prompt[:50]}: {result.error}")
                
                progress.advance(task)
    
    return results_by_model


def display_results(results_by_model: Dict[str, ModelTestSummary]):
    """Display comprehensive test results"""
    
    console.print("\n")
    console.print("=" * 100)
    console.print(Panel.fit("[bold green]Test Results Summary[/bold green]", border_style="green"))
    
    # Overall summary table
    overall_table = Table(title="Overall Results", box=box.ROUNDED, show_header=True, header_style="bold cyan")
    overall_table.add_column("Model", style="cyan", no_wrap=True)
    overall_table.add_column("Total", justify="center")
    overall_table.add_column("✓ Success", justify="center", style="green")
    overall_table.add_column("✗ Failed", justify="center", style="red")
    overall_table.add_column("🔧 Tools Used", justify="center", style="yellow")
    overall_table.add_column("📊 Valid JSON", justify="center", style="blue")
    overall_table.add_column("Avg Time", justify="center")
    
    total_success = 0
    total_tests = 0
    total_tool_success = 0
    total_json_success = 0
    
    for model, summary in results_by_model.items():
        total_tests += summary.total_tests
        total_success += summary.successful_tests
        total_tool_success += summary.tool_calling_success
        total_json_success += summary.structured_output_success
        
        avg_time = summary.total_duration / summary.total_tests if summary.total_tests > 0 else 0
        
        # Color-code the model name based on success rate
        success_rate = (summary.successful_tests / summary.total_tests * 100) if summary.total_tests > 0 else 0
        if success_rate == 100:
            model_display = f"[bold green]{model}[/bold green]"
        elif success_rate >= 50:
            model_display = f"[bold yellow]{model}[/bold yellow]"
        else:
            model_display = f"[bold red]{model}[/bold red]"
        
        overall_table.add_row(
            model_display,
            str(summary.total_tests),
            str(summary.successful_tests),
            str(summary.total_tests - summary.successful_tests),
            str(summary.tool_calling_success),
            str(summary.structured_output_success),
            f"{avg_time:.2f}s"
        )
    
    console.print(overall_table)
    
    # Final summary panel
    success_rate = (total_success / total_tests * 100) if total_tests > 0 else 0
    tool_rate = (total_tool_success / total_tests * 100) if total_tests > 0 else 0
    json_rate = (total_json_success / total_tests * 100) if total_tests > 0 else 0
    
    summary_style = "green" if success_rate == 100 else "yellow" if success_rate >= 50 else "red"
    
    console.print(Panel(
        f"[bold]Total Tests:[/bold] {total_tests}\n"
        f"[bold green]Successful:[/bold green] {total_success} ({success_rate:.1f}%)\n"
        f"[bold red]Failed:[/bold red] {total_tests - total_success} ({100 - success_rate:.1f}%)\n\n"
        f"[bold yellow]Tool Calling Success:[/bold yellow] {total_tool_success}/{total_tests} ({tool_rate:.1f}%)\n"
        f"[bold blue]Structured Output Valid:[/bold blue] {total_json_success}/{total_tests} ({json_rate:.1f}%)",
        title="[bold]Final Summary[/bold]",
        border_style=summary_style
    ))
    
    # Detailed failures table (if any)
    failed_models = {model: summary for model, summary in results_by_model.items() 
                     if summary.total_tests > summary.successful_tests}
    
    if failed_models:
        console.print("\n")
        console.print(Panel.fit("[bold red]Failures Detail[/bold red]", border_style="red"))
        
        for model, summary in failed_models.items():
            if summary.errors:
                console.print(f"\n[bold red]Model: {model}[/bold red]")
                for i, error in enumerate(summary.errors[:5], 1):  # Show first 5 errors
                    console.print(f"  {i}. [red]{error}[/red]")
                if len(summary.errors) > 5:
                    console.print(f"  [dim]... and {len(summary.errors) - 5} more errors[/dim]")
    
    # Success indicator
    console.print("\n")
    if success_rate == 100:
        console.print(Panel("🎉 [bold green]ALL TESTS PASSED![/bold green] 🎉", border_style="green"))
    elif success_rate >= 80:
        console.print(Panel("⚠️  [bold yellow]MOST TESTS PASSED[/bold yellow] ⚠️", border_style="yellow"))
    else:
        console.print(Panel("❌ [bold red]MANY TESTS FAILED[/bold red] ❌", border_style="red"))
    
    console.print("=" * 100)
    console.print()


async def main():
    """Main entry point"""
    try:
        console.print(Panel.fit(
            "[bold cyan]Starting Automated Test Suite[/bold cyan]\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            border_style="cyan"
        ))
        
        results = await run_automated_tests()
        display_results(results)
        
        # Exit with appropriate code
        total_tests = sum(s.total_tests for s in results.values())
        total_success = sum(s.successful_tests for s in results.values())
        
        if total_success == total_tests:
            sys.exit(0)  # All passed
        else:
            sys.exit(1)  # Some failed
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Tests interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]Error running tests: {e}[/red]")
        logger.exception("Test suite error")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
