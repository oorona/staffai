"""
Bot API Test Tool - Interactive testing for LLM, MCP, and bot functionality

Features:
- Interactive menu system
- Test single or multiple models
- Test single or multiple prompts (from file or interactive)
- Enable/disable MCP tools
- View Redis data structures (context, stats, rate limits)
- Use real conversation context for testing
- View raw and formatted outputs
- Test with custom configurations
- Export results
"""

import asyncio
import logging
import os
import sys
import json
import redis
import warnings
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from pathlib import Path

# Suppress noisy HTTP connection warnings and tracebacks
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*SSE.*")
warnings.filterwarnings("ignore", message=".*httpx.*")

# Set up logging to suppress MCP connection noise
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("mcp").setLevel(logging.CRITICAL)  # Completely suppress MCP library errors
logging.getLogger("mcp.client").setLevel(logging.CRITICAL)  # Suppress MCP client errors
logging.getLogger("mcp.client.streamable_http").setLevel(logging.CRITICAL)  # Suppress specific SSE errors

# Suppress verbose info messages from bot utils modules (show only warnings/errors by default)
logging.getLogger("utils.litellm_client").setLevel(logging.WARNING)
logging.getLogger("utils.message_handler").setLevel(logging.WARNING)

# Add a custom filter to block specific MCP error messages
class MCPErrorFilter(logging.Filter):
    def filter(self, record):
        # Block SSE stream errors and connection errors from MCP
        if any(x in record.getMessage().lower() for x in [
            'error reading sse stream',
            'peer closed connection',
            'remoteprotocolerror',
            'incomplete chunked read'
        ]):
            return False
        return True

# Apply the filter to all loggers that might show MCP errors
for logger_name in ['mcp', 'mcp.client', 'mcp.client.streamable_http', 'httpx', 'httpcore']:
    logging.getLogger(logger_name).addFilter(MCPErrorFilter())

# Rich for beautiful UI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax
from rich.layout import Layout
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box
from rich.tree import Tree
from rich.json import JSON

# Add parent directory to import bot modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

# Import bot components
from utils.message_handler import MessageHandler, MessageHandlerResult
from utils.litellm_client import LiteLLMClient

console = Console()

# Configure logging
# Set to WARNING to reduce verbosity. Change to INFO or DEBUG for detailed output.
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s:%(levelname)s:%(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def _read_docker_secret(secret_name: str) -> Optional[str]:
    """Read a Docker secret mounted at /run/secrets/<secret_name> if present.
    Returns the secret string (stripped) or None if not available.
    """
    secret_path = f"/run/secrets/{secret_name}"
    try:
        if os.path.exists(secret_path):
            with open(secret_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
    except Exception:
        logger.debug(f"Could not read docker secret {secret_name} at {secret_path}")
    return None


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
        
    def __eq__(self, other):
        """Allow comparison for mention detection"""
        if hasattr(other, 'id'):
            return self.id == other.id
        return False
        
    def __hash__(self):
        """Make hashable for set operations"""
        return hash(self.id)


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


class BotAPITestTool:
    """Interactive test tool for bot API calls"""
    
    def __init__(self):
        self.console = console
        self.config = self.load_config()
        self.redis_client = self.connect_redis()
        self.results_history = []
        
        # MCP Tools cache (session-level, loaded once)
        self._mcp_tools_cache = None
        self._mcp_servers_for_cache = None
        
        # Test configuration
        self.test_user_id = 123456789
        self.test_channel_id = 987654321
        self.test_guild_id = 111222333
        
        # Mock bot attributes (ones not defined as properties below)
        self.user = None  # Mock bot user (None for testing)
        self.response_chance = 1.0  # Always respond in tests
        # Note: stats_cog is defined as class attribute after properties
        # Note: random_response_delivery_chance is defined as @property
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from environment and defaults"""
        # Load models from both LITELLM_MODEL (singular) and LITELLM_MODELS (plural)
        model_str = os.getenv('LITELLM_MODEL', '')
        models_str = os.getenv('LITELLM_MODELS', '')
        
        # Combine both, prioritizing LITELLM_MODELS if set
        if models_str:
            default_models = [m.strip() for m in models_str.split(',') if m.strip()]
        elif model_str:
            default_models = [m.strip() for m in model_str.split(',') if m.strip()]
        else:
            default_models = ['gpt-4o-mini']  # Fallback
        
        # Load API key from Docker secret or env
        api_key_from_secret = _read_docker_secret('litellm_api_key')
        api_key_from_env = os.getenv('LITELLM_API_KEY', 'sk-1234')
        api_key = api_key_from_secret or api_key_from_env
        
        logger.info(f"API Key source: {'Docker secret' if api_key_from_secret else 'Environment variable'}")
        logger.info(f"API Key loaded: {api_key[:10]}..." if api_key else "No API key loaded")
        
        config = {
            'litellm_url': os.getenv('LITELLM_API_URL', 'http://localhost:4000'),
            'litellm_api_key': api_key,
            'redis_host': os.getenv('REDIS_HOST', 'localhost'),
            'redis_port': int(os.getenv('REDIS_PORT', '6379')),
            'redis_db': int(os.getenv('REDIS_DB', '0')),
            'redis_password': os.getenv('REDIS_PASSWORD'),
            'default_models': default_models,
            'mcp_servers': [s.strip() for s in os.getenv('MCP_SERVERS', '').split(',') if s.strip()],
            'prompts_file': 'prompts/test_prompts.txt',
            'max_history': int(os.getenv('MAX_HISTORY_PER_USER', '20')),
            'context_ttl': int(os.getenv('CONTEXT_HISTORY_TTL_SECONDS', '1800')),
            'context_max_age': int(os.getenv('CONTEXT_MESSAGE_MAX_AGE_SECONDS', '1800')),
        }
        
        # Load personality prompt
        prompt_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            'utils', 'prompts', 'personality_prompt.txt'
        )
        with open(prompt_path, 'r') as f:
            config['personality_prompt'] = f.read().strip()
        
        return config
    
    def connect_redis(self) -> redis.Redis:
        """Connect to Redis"""
        redis_config = {
            'host': self.config['redis_host'],
            'port': self.config['redis_port'],
            'db': self.config['redis_db'],
            'decode_responses': True
        }
        if self.config['redis_password']:
            redis_config['password'] = self.config['redis_password']
        
        # Show connection attempt details
        conn_str = f"{redis_config['host']}:{redis_config['port']}/db{redis_config['db']}"
        if redis_config.get('password'):
            conn_str += " (with password)"
        console.print(f"[cyan]Attempting Redis connection to:[/cyan] {conn_str}")
        
        try:
            client = redis.Redis(**redis_config)
            client.ping()
            console.print(f"[green]✓ Redis connected successfully[/green]")
            return client
        except Exception as e:
            console.print(f"[red]✗ Redis connection failed:[/red] {e}")
            console.print(f"[yellow]  Connection string: {conn_str}[/yellow]")
            console.print("[yellow]  Some features will be unavailable (no context history)[/yellow]")
            return None
    
    def show_header(self):
        """Display tool header"""
        console.clear()
        header = Panel.fit(
            "[bold cyan]Bot API Test Tool[/bold cyan]\n"
            "Interactive testing for LLM, MCP, and bot functionality",
            border_style="cyan"
        )
        console.print(header)
        
        # Show configuration
        config_table = Table(show_header=False, box=None, padding=(0, 1))
        config_table.add_column(style="dim")
        config_table.add_column(style="yellow")
        config_table.add_row("LiteLLM URL:", self.config['litellm_url'])
        
        # Show Redis status with connection details
        redis_conn_str = f"{self.config['redis_host']}:{self.config['redis_port']}/db{self.config['redis_db']}"
        if self.redis_client:
            redis_status = f"[green]✓ Connected[/green] ({redis_conn_str})"
        else:
            redis_status = f"[red]✗ Not Connected[/red] ({redis_conn_str})"
        config_table.add_row("Redis:", redis_status)
        
        config_table.add_row("Models:", f"{len(self.config['default_models'])} configured")
        console.print(config_table)
        console.print()
    
    def main_menu(self) -> str:
        """Display main menu and get user choice"""
        table = Table(show_header=False, box=box.ROUNDED, border_style="cyan")
        table.add_column("Option", style="cyan", width=4)
        table.add_column("Description", style="white")
        
        table.add_row("1", "🚀 Run Test (Single/Multiple Models & Prompts)")
        table.add_row("2", "📝 Manage Test Prompts")
        table.add_row("3", "🤖 Configure Models")
        table.add_row("4", "🔧 Configure MCP Servers")
        table.add_row("5", "📊 View Redis Data (Context, Stats, Rate Limits)")
        table.add_row("6", "💾 Test with Real Conversation Context")
        table.add_row("7", "📈 View Test Results History")
        table.add_row("8", "⚙️  Settings")
        table.add_row("9", "❌ Exit")
        
        console.print(table)
        console.print()
        
        choice = Prompt.ask(
            "[bold cyan]Select an option[/bold cyan]",
            choices=["1", "2", "3", "4", "5", "6", "7", "8", "9", ""],
            default=""
        )
        
        if choice == "":
            return "9"  # Treat empty as exit
        
        return choice
    
    async def run_test(self):
        """Run test with configured options"""
        self.show_header()
        console.print("[bold yellow]Run Test[/bold yellow]\n")
        
        # Select models
        models = self.select_models()
        if not models:
            return  # User chose back
        
        # Select prompts
        prompts = self.select_prompts()
        if not prompts:
            return  # User chose back
        
        # MCP configuration
        mcp_servers = self.select_mcp_servers()
        if mcp_servers is None:
            return  # User chose back
        
        # Display options (all default to True for convenience)
        show_raw = Confirm.ask("Show raw LLM output?", default=True)
        show_stats = Confirm.ask("Show token usage stats?", default=True)
        show_tool_details = Confirm.ask("Show MCP tool execution details?", default=True)
        
        # Run tests
        console.print()
        await self.execute_tests(models, prompts, mcp_servers, show_raw, show_stats, show_tool_details)
        
        console.print()
        Prompt.ask("[dim]Press Enter to continue[/dim]")
    
    def select_models(self) -> List[str]:
        """Select which models to test"""
        console.print("[bold]Select Models:[/bold]")
        console.print("1. Select from models in .env")
        console.print("2. Enter custom models")
        console.print("[dim]Press Enter to go back[/dim]")
        
        choice = Prompt.ask("Choice", choices=["1", "2", ""], default="")
        
        if choice == "":
            return None  # Signal to go back
        
        if choice == "1":
            # Show models from .env and let user select
            available_models = [m.strip() for m in self.config['default_models']]
            
            if not available_models:
                console.print("[yellow]No models configured in .env LITELLM_MODEL[/yellow]")
                return []
            
            console.print("\n[bold]Available models from .env:[/bold]")
            table = Table(show_header=False, box=box.SIMPLE)
            for i, model in enumerate(available_models, 1):
                table.add_row(f"{i}.", model)
            console.print(table)
            
            selection = Prompt.ask(
                "\nEnter numbers to test (comma-separated, e.g., 1,3 or 'all' for all)",
                default="all"
            )
            
            if selection.lower() == 'all':
                console.print(f"[green]Using all {len(available_models)} models[/green]")
                return available_models
            
            indices = [int(x.strip()) - 1 for x in selection.split(',') if x.strip().isdigit()]
            selected = [available_models[i] for i in indices if 0 <= i < len(available_models)]
            
            if selected:
                console.print(f"[green]Selected:[/green] {', '.join(selected)}")
            return selected
        
        elif choice == "2":
            models_str = Prompt.ask("Enter models (comma-separated)")
            return [m.strip() for m in models_str.split(',') if m.strip()]
    
    def select_prompts(self) -> List[str]:
        """Select which prompts to test"""
        console.print("\n[bold]Select Prompts:[/bold]")
        console.print("1. Load from file (with preview)")
        console.print("2. Enter prompt interactively")
        console.print("3. Use default test prompts")
        console.print("[dim]Press Enter to go back[/dim]")
        
        choice = Prompt.ask("Choice", choices=["1", "2", "3", ""], default="")
        
        if choice == "":
            return None  # Signal to go back
        
        if choice == "1":
            return self.load_prompts_from_file(preview=True)
        
        elif choice == "2":
            prompts = []
            console.print("\n[dim]Enter prompts (press Enter on empty line to finish)[/dim]")
            while True:
                prompt = Prompt.ask(f"Prompt #{len(prompts) + 1}", default="")
                if not prompt.strip():
                    break
                prompts.append(prompt.strip())
            
            if not prompts:
                console.print("[yellow]No prompts entered[/yellow]")
                return None
            return prompts
        
        else:  # choice == "3"
            return [
                "What is 2+2?",
                "Tell me a short joke",
                "Explain quantum computing in one sentence",
                "What's the capital of France?",
            ]
    
    def load_prompts_from_file(self, preview: bool = False) -> List[str]:
        """Load prompts from test_prompts.txt"""
        file_path = Path(__file__).parent / self.config['prompts_file']
        
        # Always show the file path being read
        console.print(f"[dim]Reading from:[/dim] [cyan]{file_path}[/cyan]")
        
        if not file_path.exists():
            console.print(f"[yellow]File not found:[/yellow] {file_path}")
            console.print("[dim]Creating default file...[/dim]")
            self.create_default_prompts_file(file_path)
        
        with open(file_path, 'r') as f:
            prompts = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        console.print(f"[green]Loaded {len(prompts)} prompts from file[/green]")
        
        # Preview first 15 prompts if requested
        if preview and prompts:
            preview_count = min(15, len(prompts))
            console.print(f"\n[bold cyan]First {preview_count} prompts:[/bold cyan]")
            for i, prompt in enumerate(prompts[:preview_count], 1):
                # Truncate long prompts for display
                display_prompt = prompt[:80] + "..." if len(prompt) > 80 else prompt
                console.print(f"  {i}. {display_prompt}")
            if len(prompts) > 15:
                console.print(f"[dim]  ... and {len(prompts) - 15} more[/dim]")
            console.print()
        
        return prompts
    
    def create_default_prompts_file(self, file_path: Path):
        """Create default test_prompts.txt"""
        default_prompts = """# Test Prompts for Bot API Test Tool
# One prompt per line, # for comments

# Basic tests
What is 2+2?
Tell me a joke
What's the weather like today?

# MCP tool tests (if enabled)
What files are in the current directory?
Search for information about Python asyncio
Show me a cat gif

# Structured output tests
Calculate the integral of e^x
Write a simple Python function to sort a list
What's 15% of 200?
"""
        with open(file_path, 'w') as f:
            f.write(default_prompts)
        console.print(f"[green]Created:[/green] {file_path}")
    
    def select_mcp_servers(self) -> List[str]:
        """Select which MCP servers to use"""
        console.print("\n[bold]Configure MCP Servers:[/bold]")
        console.print("1. Use all servers from .env")
        console.print("2. Select specific servers from .env")
        console.print("3. Disable MCP (no tools)")
        console.print("4. Enter custom MCP servers")
        console.print("[dim]Press Enter to go back[/dim]")
        
        choice = Prompt.ask("Choice", choices=["1", "2", "3", "4", ""], default="")
        
        if choice == "":
            return None  # Signal to go back
        
        if choice == "1":
            # Use all MCP servers from config
            servers = self.config['mcp_servers']
            if servers:
                console.print(f"[green]Using all {len(servers)} MCP servers from .env[/green]")
            else:
                console.print("[yellow]No MCP servers configured in .env[/yellow]")
            return servers
        
        elif choice == "2":
            # Let user select specific servers
            available_servers = self.config['mcp_servers']
            
            if not available_servers:
                console.print("[yellow]No MCP servers configured in .env[/yellow]")
                return []
            
            console.print("\n[bold]Available MCP servers from .env:[/bold]")
            table = Table(show_header=False, box=box.SIMPLE)
            for i, server in enumerate(available_servers, 1):
                # Show short name if URL
                display_name = server.split('/')[-2] if '/' in server else server
                table.add_row(f"{i}.", f"{display_name} ({server})")
            console.print(table)
            
            selection = Prompt.ask(
                "\nEnter numbers to use (comma-separated, e.g., 1,3 or 'all' for all)",
                default="all"
            )
            
            if selection.lower() == 'all':
                console.print(f"[green]Using all {len(available_servers)} servers[/green]")
                return available_servers
            
            indices = [int(x.strip()) - 1 for x in selection.split(',') if x.strip().isdigit()]
            selected = [available_servers[i] for i in indices if 0 <= i < len(available_servers)]
            
            if selected:
                console.print(f"[green]Selected {len(selected)} servers[/green]")
            return selected
        
        elif choice == "3":
            # Disable MCP
            console.print("[yellow]MCP tools disabled[/yellow]")
            return []
        
        else:  # choice == "4"
            # Custom MCP servers
            servers_str = Prompt.ask("Enter MCP server URLs (comma-separated)")
            servers = [s.strip() for s in servers_str.split(',') if s.strip()]
            if servers:
                console.print(f"[green]Using {len(servers)} custom servers[/green]")
            return servers
    
    async def execute_tests(
        self, 
        models: List[str], 
        prompts: List[str],
        mcp_servers: List[str],
        show_raw: bool,
        show_stats: bool,
        show_tool_details: bool = True
    ):
        """Execute tests with progress tracking"""
        total_tests = len(models) * len(prompts)
        current_test = 0
        
        results = []
        
        # Create a shared LiteLLM client pool (one per model) to reuse MCP tool cache
        shared_clients = {}
        
        # Preload MCP tools ONCE for all tests in this session (check cache first)
        if mcp_servers:
            # Check if we need to reload (different servers or not cached yet)
            servers_changed = self._mcp_servers_for_cache != mcp_servers
            
            if self._mcp_tools_cache is None or servers_changed:
                if servers_changed and self._mcp_tools_cache:
                    console.print(f"[yellow]⚠️  MCP server list changed, reloading tools...[/yellow]")
                
                console.print(f"\n[cyan]🔧 Preloading MCP tools from {len(mcp_servers)} servers...[/cyan]")
                try:
                    # Create a client just for fetching tools
                    preload_client = LiteLLMClient(
                        model=models[0],
                        base_url=self.config['litellm_url'],
                        api_key=self.config['litellm_api_key'],
                        redis_client=self.redis_client,
                        mcp_servers=mcp_servers
                    )
                    preload_client.show_tool_details = True  # Enable pretty tool display
                    mcp_tools = await preload_client.get_mcp_tools()
                    if mcp_tools:
                        # Cache for future tests
                        self._mcp_tools_cache = mcp_tools
                        self._mcp_servers_for_cache = mcp_servers
                        console.print(f"[green]✅ Cached {len(mcp_tools)} MCP tools for session[/green]")
                    else:
                        console.print("[yellow]⚠️  No MCP tools loaded[/yellow]")
                except Exception as e:
                    console.print(f"[red]❌ Failed to preload MCP tools: {e}[/red]")
                    logger.error(f"Failed to preload MCP tools: {e}")  # Removed .exception() to avoid stack trace
            else:
                console.print(f"[green]✅ Using cached {len(self._mcp_tools_cache)} MCP tools from session[/green]")
            
            # Create shared clients with cached tools
            if self._mcp_tools_cache:
                import time
                for model in models:
                    client = LiteLLMClient(
                        model=model,
                        base_url=self.config['litellm_url'],
                        api_key=self.config['litellm_api_key'],
                        redis_client=self.redis_client,
                        context_history_ttl_seconds=self.config['context_ttl'],
                        context_message_max_age_seconds=self.config['context_max_age'],
                        max_history_messages=self.config['max_history'],
                        mcp_servers=mcp_servers
                    )
                    # Inject preloaded tools
                    client._mcp_tools_cache = self._mcp_tools_cache
                    client._mcp_tools_cache_time = time.time()
                    shared_clients[model] = client
                
                console.print(f"[green]✅ Created {len(shared_clients)} shared clients with cached tools[/green]")
        
        # If no MCP servers or preload failed, create clients without tools
        for model in models:
            if model not in shared_clients:
                shared_clients[model] = LiteLLMClient(
                    model=model,
                    base_url=self.config['litellm_url'],
                    api_key=self.config['litellm_api_key'],
                    redis_client=self.redis_client,
                    context_history_ttl_seconds=self.config['context_ttl'],
                    context_message_max_age_seconds=self.config['context_max_age'],
                    max_history_messages=self.config['max_history'],
                    mcp_servers=mcp_servers if mcp_servers else None
                )
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"[cyan]Running {total_tests} tests...", total=total_tests)
            
            for model in models:
                for prompt in prompts:
                    current_test += 1
                    progress.update(
                        task, 
                        advance=1,
                        description=f"[cyan]Test {current_test}/{total_tests}: {model[:30]}..."
                    )
                    
                    result = await self.execute_single_test(
                        model, prompt, show_raw, show_stats, show_tool_details, shared_clients[model]
                    )
                    results.append(result)
        
        # Display results
        self.display_results(results, show_raw, show_stats, show_tool_details)
        
        # Store in history
        self.results_history.append({
            'timestamp': datetime.now().isoformat(),
            'models': models,
            'prompts': prompts,
            'mcp_enabled': bool(mcp_servers),
            'results': results
        })
    
    async def execute_single_test(
        self,
        model: str,
        prompt: str,
        show_raw: bool,
        show_stats: bool,
        show_tool_details: bool,
        shared_client: LiteLLMClient
    ) -> Dict[str, Any]:
        """Execute a single test using a shared LiteLLM client"""
        try:
            # Use the shared client (has cached MCP tools)
            self.litellm_client = shared_client
            
            # Enable/disable tool details based on user choice
            self.litellm_client.show_tool_details = show_tool_details
            
            # Create message handler (it will use bot.litellm_client and bot.redis_client)
            message_handler = MessageHandler(bot=self)
            
            # Create mock message
            mock_msg = MockMessage(
                content=prompt,
                user_id=self.test_user_id,
                channel_id=self.test_channel_id,
                guild_id=self.test_guild_id
            )
            
            # Execute
            start_time = datetime.now()
            result = await message_handler.handle_message(message=mock_msg)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return {
                'model': model,
                'prompt': prompt,
                'success': result['should_respond'],
                'response_type': result.get('response_type'),
                'response_text': result.get('response_text'),
                'response_data': result.get('response_data'),
                'raw_output': result.get('raw_output'),
                'tokens_used': result.get('tokens_used'),
                'prompt_tokens': result.get('prompt_tokens'),
                'completion_tokens': result.get('completion_tokens'),
                'cached_tokens': result.get('cached_tokens'),
                'reasoning_tokens': result.get('reasoning_tokens'),
                'history_length': result.get('history_length', 0),
                'messages_sent': result.get('messages_sent', 0),
                'llm_calls': result.get('llm_calls', []),
                'duration': duration,
                'error': result.get('error')
            }
        
        except Exception as e:
            logger.error(f"Test failed for {model}: {e}")  # Removed .exception() to avoid stack trace
            return {
                'model': model,
                'prompt': prompt,
                'success': False,
                'error': str(e),
                'duration': 0
            }
    
    def display_results(self, results: List[Dict], show_raw: bool, show_stats: bool, show_tool_details: bool = False):
        """Display test results"""
        console.print("\n" + "="*80)
        console.print("[bold green]Test Results[/bold green]")
        console.print("="*80 + "\n")
        
        for i, result in enumerate(results, 1):
            # Create panel for each result
            if result['success']:
                content = f"[bold]Model:[/bold] {result['model']}\n"
                content += f"[bold]Prompt:[/bold] {result['prompt']}\n\n"
                content += f"[bold cyan]Response Type:[/bold cyan] {result['response_type']}\n"
                content += f"[bold cyan]Response:[/bold cyan]\n{result['response_text']}\n"
                
                if result['response_data']:
                    content += f"\n[bold cyan]Data:[/bold cyan]\n{result['response_data']}\n"
                
                if show_stats and result.get('tokens_used'):
                    content += f"\n[bold yellow]Token Usage:[/bold yellow]\n"
                    content += f"  Total: {result['tokens_used']} tokens\n"
                    if result.get('prompt_tokens'):
                        cached = result.get('cached_tokens', 0)
                        content += f"  Prompt: {result['prompt_tokens']} tokens (Cached: {cached})"
                        # Show context info
                        history_len = result.get('history_length', 0)
                        messages_sent = result.get('messages_sent', 0)
                        if history_len > 0 or messages_sent > 0:
                            content += f" [dim][History: {history_len} msgs, Sent: {messages_sent} msgs][/dim]"
                        content += "\n"
                    if result.get('completion_tokens'):
                        content += f"  Completion: {result['completion_tokens']} tokens"
                        if result.get('reasoning_tokens'):
                            content += f" (Reasoning: {result['reasoning_tokens']})"
                        content += "\n"
                    
                    # Calculate estimated cost (rough estimates)
                    cost = self._estimate_cost(
                        result['model'],
                        result.get('prompt_tokens', 0),
                        result.get('completion_tokens', 0),
                        result.get('cached_tokens', 0)
                    )
                    if cost is not None:
                        content += f"  Estimated Cost: ${cost:.6f}\n"
                    else:
                        content += f"  [dim]Cost: Unknown (model: {result['model']})[/dim]\n"
                    
                    content += f"  Duration: {result['duration']:.2f}s"
                
                # Show LLM call breakdown if available (tool calling passes)
                if show_stats and result.get('llm_calls'):
                    llm_calls = result['llm_calls']
                    if len(llm_calls) > 1:
                        content += f"\n\n[bold cyan]LLM Call Breakdown:[/bold cyan]\n"
                        for call in llm_calls:
                            content += f"  Pass {call['pass_number']}: {call['purpose']}\n"
                            if call.get('tool_calls'):
                                content += f"    Tools: {', '.join([tc['name'] for tc in call['tool_calls']])}\n"
                            if call.get('tokens'):
                                content += f"    Tokens: {call['tokens']['total']} "
                                content += f"(prompt: {call['tokens']['prompt']}, completion: {call['tokens']['completion']})\n"
                            content += f"    Duration: {call['duration']:.2f}s\n"
                
                if show_raw and result.get('raw_output'):
                    content += f"\n\n[bold yellow]Raw Output:[/bold yellow]\n"
                    content += f"[dim]{result['raw_output']}[/dim]"
                
                panel = Panel(
                    content,
                    title=f"Test #{i}",
                    border_style="green"
                )
            else:
                content = f"[bold]Model:[/bold] {result['model']}\n"
                content += f"[bold]Prompt:[/bold] {result['prompt']}\n\n"
                content += f"[bold red]Error:[/bold red] {result.get('error', 'Unknown error')}"
                
                panel = Panel(
                    content,
                    title=f"Test #{i} - FAILED",
                    border_style="red"
                )
            
            console.print(panel)
            console.print()
    
    def _estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0) -> Optional[float]:
        """
        Estimate cost based on model and token usage.
        Uses manual pricing calculation based on known rates.
        """
        # Handle None values
        prompt_tokens = prompt_tokens or 0
        completion_tokens = completion_tokens or 0
        cached_tokens = cached_tokens or 0
        
        if prompt_tokens == 0 and completion_tokens == 0:
            return None
        
        # Pricing table (input/output/cached per 1M tokens)
        pricing = {
            # OpenAI models
            'gpt-5-mini': (0.15, 0.60, 0.075),  # Estimated similar to gpt-4o-mini
            'gpt-5-nano': (0.05, 0.20, 0.025),  # Estimated even cheaper
            'gpt-4o': (2.50, 10.00, 1.25),
            'gpt-4o-mini': (0.150, 0.600, 0.075),
            'gpt-4-turbo': (10.00, 30.00, 5.00),
            'gpt-4': (30.00, 60.00, 15.00),
            'gpt-3.5-turbo': (0.50, 1.50, 0.25),
            'o1-preview-2024-09-12': (15.00, 60.00, 7.50),
            'o1-preview': (15.00, 60.00, 7.50),
            'o1-mini-2024-09-12': (3.00, 12.00, 1.50),
            'o1-mini': (3.00, 12.00, 1.50),
            'o1': (15.00, 60.00, 7.50),
            
            # xAI models
            'grok-3-mini': (0.50, 1.50, 0.25),  # Estimated similar to gpt-3.5-turbo
            'grok-3-mini-fast': (0.30, 0.90, 0.15),  # Faster = cheaper
            
            # Anthropic models
            'claude-3-5-sonnet': (3.00, 15.00, 1.50),
            'claude-3-opus': (15.00, 75.00, 7.50),
            'claude-3-sonnet': (3.00, 15.00, 1.50),
            'claude-3-haiku': (0.25, 1.25, 0.125),
            
            # Google models (Gemini)
            'gemini-2.5-flash-lite': (0.05, 0.20, 0.025),  # Lite = very cheap
            'gemini-2.5-flash': (0.10, 0.40, 0.05),
            'gemini-2.5': (0.10, 0.40, 0.05),
            'gemini-2.0': (0.10, 0.40, 0.05),
            'gemini-1.5-pro': (1.25, 5.00, 0.625),
            'gemini-1.5-flash': (0.075, 0.30, 0.0375),
            'gemini-pro': (0.50, 1.50, 0.25),
            'gemini-flash': (0.075, 0.30, 0.0375),
            
            # Meta models
            'llama-3': (0.10, 0.10, 0.05),
            'llama-2': (0.20, 0.20, 0.10),
            
            # Mistral models
            'mistral-large': (3.00, 9.00, 1.50),
            'mistral-medium': (2.70, 8.10, 1.35),
            'mistral-small': (1.00, 3.00, 0.50),
            'mistral-tiny': (0.25, 0.25, 0.125),
            
            # DeepSeek models
            'deepseek': (0.14, 0.28, 0.07),
            
            # Groq models (typically free or very cheap)
            'groq': (0.05, 0.10, 0.025),
        }
        
        model_lower = model.lower()
        matched_pricing = None
        matched_key = None
        
        # Try to find matching pricing
        for model_key, prices in pricing.items():
            if model_key in model_lower:
                matched_pricing = prices
                matched_key = model_key
                break
        
        if not matched_pricing:
            logger.debug(f"No pricing data for model: {model}")
            return None
        
        input_price, output_price, cached_price = matched_pricing
        uncached_prompt = prompt_tokens - cached_tokens
        
        cost = (uncached_prompt * input_price / 1_000_000 +
                cached_tokens * cached_price / 1_000_000 +
                completion_tokens * output_price / 1_000_000)
        
        logger.debug(f"Cost for {model} (matched: {matched_key}): ${cost:.6f} "
                    f"(prompt: {prompt_tokens}, completion: {completion_tokens}, cached: {cached_tokens})")
        
        return cost
    
    def view_redis_data(self):
        """View Redis data structures"""
        if not self.redis_client:
            console.print("[red]Redis not connected[/red]")
            Prompt.ask("[dim]Press Enter to continue[/dim]")
            return
        
        while True:
            self.show_header()
            console.print("[bold yellow]Redis Data Viewer[/bold yellow]\n")
            
            table = Table(show_header=False, box=box.ROUNDED)
            table.add_row("1", "View Conversation Context")
            table.add_row("2", "View Token Stats")
            table.add_row("3", "View Rate Limits")
            table.add_row("4", "View All Keys (by pattern)")
            table.add_row("", "[dim]Press Enter to go back[/dim]")
            
            console.print(table)
            choice = Prompt.ask("Select", choices=["1", "2", "3", "4", ""], default="")
            
            if choice == "":
                break
            elif choice == "1":
                self.view_conversation_context()
            elif choice == "2":
                self.view_token_stats()
            elif choice == "3":
                self.view_rate_limits()
            elif choice == "4":
                pattern = Prompt.ask("Enter key pattern", default="*")
                self.view_keys_by_pattern(pattern)
    
    def paginate_selection(self, items: List[Any], page_size: int = 20, 
                          title: str = "Select Item",
                          format_func = None) -> Optional[Any]:
        """
        Generic pagination helper for selecting from large lists.
        
        Args:
            items: List of items to paginate
            page_size: Number of items per page
            title: Title for the selection
            format_func: Optional function to format item display (item, index) -> dict with column values
        
        Returns:
            Selected item or None if cancelled
        """
        if not items:
            return None
        
        total_pages = (len(items) + page_size - 1) // page_size
        current_page = 0
        
        while True:
            start_idx = current_page * page_size
            end_idx = min(start_idx + page_size, len(items))
            page_items = items[start_idx:end_idx]
            
            console.print(f"\n[bold cyan]{title}[/bold cyan]")
            console.print(f"[dim]Total: {len(items)} items | Page {current_page + 1}/{total_pages}[/dim]\n")
            
            # Display items
            table = Table(box=box.ROUNDED)
            table.add_column("#", style="yellow", width=5)
            
            # Add columns based on format_func or default
            if format_func:
                # Get sample to determine columns
                sample = format_func(items[0], 0) if items else {}
                for col_name in sample.keys():
                    table.add_column(col_name, style="cyan" if col_name != "Key" else "dim")
                
                for i, item in enumerate(page_items):
                    global_idx = start_idx + i + 1
                    formatted = format_func(item, global_idx - 1)
                    table.add_row(str(global_idx), *formatted.values())
            else:
                table.add_column("Item", style="cyan")
                for i, item in enumerate(page_items):
                    global_idx = start_idx + i + 1
                    table.add_row(str(global_idx), str(item))
            
            console.print(table)
            
            # Navigation options
            console.print("\n[bold]Options:[/bold]")
            if current_page > 0:
                console.print("  [cyan]p[/cyan] - Previous page")
            if current_page < total_pages - 1:
                console.print("  [cyan]n[/cyan] - Next page")
            console.print("  [cyan]1-{}[/cyan] - Select item".format(len(items)))
            console.print("  [dim]Press Enter to cancel[/dim]")
            
            choice = Prompt.ask("\nSelection", default="").strip().lower()
            
            if choice == "":
                return None
            elif choice == "n" and current_page < total_pages - 1:
                current_page += 1
            elif choice == "p" and current_page > 0:
                current_page -= 1
            elif choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(items):
                    return items[idx]
                else:
                    console.print(f"[red]Invalid selection. Choose 1-{len(items)}[/red]")
            else:
                console.print("[red]Invalid option[/red]")
    
    def get_available_user_ids(self) -> List[tuple]:
        """Get list of available user IDs from Redis with their channel IDs"""
        pattern = "discord_context:*"
        keys = list(self.redis_client.scan_iter(match=pattern, count=500))
        
        user_channels = []
        for key in keys:
            # Parse key: discord_context:{user_id}:{channel_id}
            parts = key.split(':')
            if len(parts) == 3:
                user_id = parts[1]
                channel_id = parts[2]
                user_channels.append((user_id, channel_id, key))
        
        return user_channels
    
    def view_conversation_context(self):
        """View conversation context from Redis"""
        console.print("\n[bold]Conversation Context Viewer[/bold]\n")
        
        # Get available user IDs
        user_channels = self.get_available_user_ids()
        
        if not user_channels:
            console.print("[yellow]No conversation contexts found in Redis[/yellow]")
            console.print("[dim]Hint: Try using the bot first to create some context[/dim]")
            Prompt.ask("\n[dim]Press Enter to continue[/dim]")
            return
        
        # Use pagination to select context
        def format_context(item, idx):
            user_id, channel_id, key = item
            return {
                "User ID": user_id,
                "Channel ID": channel_id,
                "Key": key
            }
        
        selected = self.paginate_selection(
            user_channels,
            page_size=20,
            title=f"Conversation Contexts ({len(user_channels)} found)",
            format_func=format_context
        )
        
        if not selected:
            return
        
        user_id, channel_id, key = selected
        
        try:
            context = self.redis_client.get(key)
            if context:
                context_data = json.loads(context)
                
                console.print(f"\n[bold cyan]Context for User {user_id}, Channel {channel_id}:[/bold cyan]")
                console.print(f"[dim]Messages: {len(context_data)}[/dim]\n")
                console.print(JSON(json.dumps(context_data, indent=2)))
                
                # Offer to use this context
                if Confirm.ask("\nUse this context for a test?"):
                    self.test_with_context(context_data, user_id, channel_id)
            else:
                console.print(f"[yellow]No context found for key:[/yellow] {key}")
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
        
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
    
    def view_token_stats(self):
        """View token consumption stats"""
        console.print("\n[bold]Token Stats Viewer[/bold]\n")
        
        pattern = "token_stats:*"
        keys = list(self.redis_client.scan_iter(match=pattern, count=500))
        
        if not keys:
            console.print("[yellow]No token stats found[/yellow]")
            Prompt.ask("\n[dim]Press Enter to continue[/dim]")
            return
        
        # Prepare data with values
        stats_data = []
        for key in keys:
            value = self.redis_client.get(key)
            stats_data.append((key, str(value)))
        
        # Use pagination
        def format_stat(item, idx):
            key, value = item
            return {
                "Key": key,
                "Value": value
            }
        
        self.paginate_selection(
            stats_data,
            page_size=25,
            title=f"Token Stats ({len(stats_data)} entries)",
            format_func=format_stat
        )
        
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
    
    def view_rate_limits(self):
        """View rate limit data"""
        console.print("\n[bold]Rate Limits Viewer[/bold]\n")
        
        pattern = "*_rl:*"
        keys = list(self.redis_client.scan_iter(match=pattern, count=500))
        
        if not keys:
            console.print("[yellow]No rate limit data found[/yellow]")
            Prompt.ask("\n[dim]Press Enter to continue[/dim]")
            return
        
        # Prepare data with parsed information
        rate_limit_data = []
        for key in keys:
            # Parse key type (msg_rl or token_rl)
            key_type = "Message" if 'msg_rl' in key else "Token" if 'token_rl' in key else "Other"
            
            # Extract guild:user from key
            parts = key.split(':')
            if len(parts) >= 3:
                guild_user = f"{parts[1]}:{parts[2]}"
            else:
                guild_user = key
            
            count = self.redis_client.llen(key) if 'msg_rl' in key or 'token_rl' in key else "N/A"
            rate_limit_data.append((key, key_type, guild_user, str(count)))
        
        # Use pagination
        def format_rate_limit(item, idx):
            key, key_type, guild_user, count = item
            return {
                "Type": key_type,
                "Guild:User": guild_user,
                "Count": count,
                "Key": key
            }
        
        self.paginate_selection(
            rate_limit_data,
            page_size=25,
            title=f"Rate Limits ({len(rate_limit_data)} entries)",
            format_func=format_rate_limit
        )
        
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
    
    def view_keys_by_pattern(self, pattern: str):
        """View Redis keys by pattern"""
        console.print(f"\n[bold]Keys matching:[/bold] {pattern}\n")
        
        keys = list(self.redis_client.scan_iter(match=pattern, count=1000))
        
        if not keys:
            console.print("[yellow]No keys found[/yellow]")
            Prompt.ask("\n[dim]Press Enter to continue[/dim]")
            return
        
        # Use pagination for key browsing
        def format_key(item, idx):
            return {
                "Key": item
            }
        
        self.paginate_selection(
            keys,
            page_size=30,
            title=f"Keys matching '{pattern}' ({len(keys)} found)",
            format_func=format_key
        )
        
        Prompt.ask("\n[dim]Press Enter to continue[/dim]")
    
    async def test_with_context(self, context_data: List[Dict], user_id: str, channel_id: str):
        """Test using existing conversation context"""
        console.print("\n[bold]Test with Existing Context[/bold]\n")
        
        # Show context summary
        console.print(f"[dim]Context has {len(context_data)} messages[/dim]\n")
        
        prompt = Prompt.ask("Enter your test prompt")
        model = Prompt.ask("Model", default=self.config['default_models'][0])
        
        # Execute test (context will be loaded from Redis automatically)
        result = await self.execute_single_test(
            model=model,
            prompt=prompt,
            mcp_servers=self.config['mcp_servers'],
            show_raw=True,
            show_stats=True
        )
        
        self.display_results([result], show_raw=True, show_stats=True)
    
    # Mock bot attributes
    @property
    def personality_prompt(self):
        return self.config['personality_prompt']
    
    @property
    def chat_system_prompt(self):
        return self.config['personality_prompt']
    
    @property
    def ignored_role_ids_set(self):
        return set()
    
    @property
    def super_role_ids_set(self):
        return set()
    
    @property
    def rate_limit_count(self):
        return 999999
    
    @property
    def rate_limit_window_seconds(self):
        return 60
    
    @property
    def token_rate_limit_count(self):
        return 999999
    
    @property
    def restricted_user_role_id(self):
        return None
    
    @property
    def restricted_channel_id(self):
        return None
    
    @property
    def rate_limit_message_user_template(self):
        return ""
    
    @property
    def restricted_channel_message_user_template(self):
        return ""
    
    @property
    def restriction_duration_seconds(self):
        return 0
    
    @property
    def random_response_delivery_chance(self):
        return 1.0
    
    stats_cog = None
    
    async def run(self):
        """Main run loop"""
        # Give user a moment to see Redis connection status
        console.print("\n[dim]Starting test tool...[/dim]")
        await asyncio.sleep(1.5)
        
        while True:
            self.show_header()
            choice = self.main_menu()
            
            if choice == "1":
                await self.run_test()
            elif choice == "2":
                console.print("[yellow]Feature coming soon[/yellow]")
                Prompt.ask("[dim]Press Enter[/dim]")
            elif choice == "3":
                console.print("[yellow]Feature coming soon[/yellow]")
                Prompt.ask("[dim]Press Enter[/dim]")
            elif choice == "4":
                console.print("[yellow]Feature coming soon[/yellow]")
                Prompt.ask("[dim]Press Enter[/dim]")
            elif choice == "5":
                self.view_redis_data()
            elif choice == "6":
                self.view_conversation_context()
            elif choice == "7":
                console.print("[yellow]Feature coming soon[/yellow]")
                Prompt.ask("[dim]Press Enter[/dim]")
            elif choice == "8":
                console.print("[yellow]Feature coming soon[/yellow]")
                Prompt.ask("[dim]Press Enter[/dim]")
            elif choice == "9":
                console.print("\n[cyan]Goodbye![/cyan]\n")
                break


async def main():
    """Entry point"""
    try:
        tool = BotAPITestTool()
        await tool.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        logger.error(f"Tool error: {e}")  # Removed .exception() to avoid stack trace
        raise


if __name__ == "__main__":
    asyncio.run(main())
