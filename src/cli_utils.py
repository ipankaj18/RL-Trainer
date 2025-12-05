#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI Utilities for RL Trainer
Shared helper functions for terminal output, user input, and subprocess management.
"""
import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from datetime import datetime

# Try to import colorama for colored output
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class Colors:
    """Color constants for terminal output."""
    if COLORAMA_AVAILABLE:
        RED = Fore.RED
        GREEN = Fore.GREEN
        BLUE = Fore.BLUE
        YELLOW = Fore.YELLOW
        MAGENTA = Fore.MAGENTA
        CYAN = Fore.CYAN
        WHITE = Fore.WHITE
        BOLD = Style.BRIGHT
        RESET = Style.RESET_ALL
    else:
        RED = "\033[31m"
        GREEN = "\033[32m"
        BLUE = "\033[34m"
        YELLOW = "\033[33m"
        MAGENTA = "\033[35m"
        CYAN = "\033[36m"
        WHITE = "\033[37m"
        BOLD = "\033[1m"
        RESET = "\033[0m"


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header(title: str, color: str = Colors.CYAN):
    """Print a styled header box."""
    print(f"\n{Colors.BOLD}{color}╔{'═' * (len(title) + 4)}╗{Colors.RESET}")
    print(f"{Colors.BOLD}{color}║  {title}  ║{Colors.RESET}")
    print(f"{Colors.BOLD}{color}╚{'═' * (len(title) + 4)}╝{Colors.RESET}")


def get_user_input(prompt: str, valid_options: Optional[List[str]] = None) -> Optional[str]:
    """
    Get and validate user input.
    
    Args:
        prompt: The prompt to display to the user
        valid_options: List of valid input options
        
    Returns:
        Validated user input or None if cancelled
    """
    while True:
        try:
            user_input = input(f"{Colors.BLUE}{prompt}{Colors.RESET}").strip()
            
            if valid_options and user_input not in valid_options:
                print(f"{Colors.RED}Invalid option. Please choose from: {', '.join(valid_options)}{Colors.RESET}")
                continue

            return user_input

        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Operation cancelled by user.{Colors.RESET}")
            return None
        except EOFError:
            print(f"\n{Colors.YELLOW}Input stream ended.{Colors.RESET}")
            return None


def prompt_confirm(message: str, default_yes: bool = False) -> bool:
    """
    Prompt user for yes/no confirmation.
    
    Args:
        message: The question to ask
        default_yes: Whether 'y' is the default
        
    Returns:
        True if confirmed, False otherwise
    """
    options = ["y", "n", "Y", "N"]
    prompt_suffix = " (Y/n)" if default_yes else " (y/n)"
    
    response = get_user_input(f"{Colors.YELLOW}{message}{prompt_suffix}: {Colors.RESET}", options)
    
    if response is None:
        return False
        
    return response.lower() == 'y'


def prompt_choice(title: str, options: Dict[str, str], prompt_msg: str = "Select option") -> Optional[str]:
    """
    Display a menu of options and get user choice.
    
    Args:
        title: Menu title
        options: Dictionary of key -> description
        prompt_msg: Prompt message
        
    Returns:
        Selected key or None if cancelled
    """
    print(f"\n{Colors.BOLD}{title}:{Colors.RESET}")
    for key, desc in options.items():
        print(f"{Colors.CYAN}  {key}. {desc}{Colors.RESET}")
        
    valid_keys = list(options.keys())
    return get_user_input(f"{Colors.YELLOW}{prompt_msg} ({'/'.join(valid_keys)}): {Colors.RESET}", valid_keys)


def run_command_with_progress(command: List[str], description: str,
                          log_file: Optional[str] = None,
                          cwd: Optional[Path] = None,
                          env_vars: Optional[Dict[str, str]] = None,
                          interactive: bool = False) -> Tuple[bool, str]:
    """
    Run a command with progress indication and logging.

    Args:
        command: Command to execute as list of strings
        description: Description of the operation
        log_file: Log file to save output (relative to logs/ dir if simple name)
        cwd: Working directory
        env_vars: Environment variables to add/override
        interactive: If True, allows user input (stdin) but captures less output

    Returns:
        Tuple of (success, output)
    """
    print(f"\n{Colors.YELLOW}{description}{Colors.RESET}")
    print(f"{Colors.CYAN}Executing: {' '.join(command)}{Colors.RESET}")
    print(f"{Colors.MAGENTA}{'=' * 60}{Colors.RESET}")

    try:
        # Default to current working directory if not specified
        working_directory = cwd if cwd else Path.cwd()
        
        # Prepare environment
        env = os.environ.copy()
        
        # Ensure project root is in PYTHONPATH
        # Assuming this script is in src/ and project root is parent of src/
        # Or if called from main.py in root, cwd is root
        project_root = working_directory
        pythonpath = str(project_root)
        
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = f"{pythonpath}{os.pathsep}{env['PYTHONPATH']}"
        else:
            env['PYTHONPATH'] = pythonpath
            
        if env_vars:
            env.update(env_vars)

        output_lines = []

        # Interactive mode: allow stdin, minimal output capture
        if interactive:
            # Run interactively - user sees all output and can provide input
            process = subprocess.Popen(
                command,
                stdin=None,  # Inherit stdin from parent (allows user input)
                stdout=None,  # Inherit stdout (output goes directly to terminal)
                stderr=None,  # Inherit stderr
                cwd=working_directory,
                env=env
            )
            process.wait()
            success = process.returncode == 0
            output_lines = ["[Interactive mode - output not captured]"]

        # Non-interactive mode: capture output with progress indication
        else:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                encoding='utf-8',
                errors='replace',
                cwd=working_directory,
                env=env
            )

            # Read output line by line
            if TQDM_AVAILABLE:
                with tqdm(desc="Streaming Output", total=0, bar_format="{desc}", leave=False) as progress_bar:
                    for line in iter(process.stdout.readline, ''):
                        if not line:
                            break
                        cleaned = line.rstrip()
                        if cleaned:
                            output_lines.append(cleaned)
                            progress_bar.write(f"{Colors.WHITE}{cleaned}{Colors.RESET}")
                process.wait()
            else:
                output, _ = process.communicate()
                output_lines = output.split('\n') if output else []
                for line in output_lines:
                    cleaned = line.strip()
                    if cleaned:
                        print(f"{Colors.WHITE}{cleaned}{Colors.RESET}")

            success = process.returncode == 0
        
        if success:
            print(f"\n{Colors.GREEN}✓ {description} completed successfully!{Colors.RESET}")
        else:
            print(f"\n{Colors.RED}✗ {description} failed with return code {process.returncode}{Colors.RESET}")
        
        # Save to log file if specified
        if log_file:
            # If log_file is just a name, put it in logs/
            log_path = Path(log_file)
            if not log_path.is_absolute() and len(log_path.parts) == 1:
                log_dir = working_directory / "logs"
                log_dir.mkdir(exist_ok=True)
                log_path = log_dir / log_file
                
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(f"\n{'=' * 60}\n")
                f.write(f"{datetime.now().isoformat()} - {description}\n")
                f.write(f"Command: {' '.join(command)}\n")
                f.write(f"Return Code: {process.returncode}\n")
                output_text = '\n'.join(output_lines)
                f.write(f"Output:\n{output_text}\n")
        
        return success, '\n'.join(output_lines)
        
    except subprocess.TimeoutExpired:
        print(f"\n{Colors.RED}✗ {description} timed out{Colors.RESET}")
        return False, "Operation timed out"
    except Exception as e:
        print(f"\n{Colors.RED}✗ {description} failed: {str(e)}{Colors.RESET}")
        return False, str(e)


def detect_and_select_market(project_dir: Path) -> Optional[str]:
    """
    Detect available market data files and prompt user to select one.

    Args:
        project_dir: Project root directory

    Returns:
        Selected market symbol (e.g., 'NQ', 'ES') or None if cancelled
    """
    try:
        from src.market_specs import get_market_spec

        data_dir = project_dir / "data"
        if not data_dir.exists():
            print(f"\n{Colors.RED}Data directory not found: {data_dir}{Colors.RESET}")
            return None

        # Find all *_D1M.csv files
        minute_files = list(data_dir.glob("*_D1M.csv"))

        if not minute_files:
            print(f"\n{Colors.RED}No market data files found in data/ directory.{Colors.RESET}")
            print(f"{Colors.YELLOW}Please run 'Data Processing' first to prepare market data.{Colors.RESET}")
            return None

        # Extract market symbols
        available_markets = []
        for file in minute_files:
            market = file.stem.replace("_D1M", "")
            available_markets.append({
                'market': market,
                'minute_file': file.name
            })

        if not available_markets:
            print(f"\n{Colors.RED}No market data files found in data/ directory.{Colors.RESET}")
            print(f"{Colors.YELLOW}Please run 'Data Processing' first to prepare market data.{Colors.RESET}")
            return None

        print_header("MARKET SELECTION")
        print()

        # If only one market, auto-select it
        if len(available_markets) == 1:
            market = available_markets[0]['market']
            market_spec = get_market_spec(market)
            spec_info = f"${market_spec.contract_multiplier} x {market_spec.tick_size} tick = ${market_spec.tick_value:.2f}"

            print(f"{Colors.GREEN}Detected 1 market:{Colors.RESET}")
            print(f"  • {Colors.BOLD}{market}{Colors.RESET} - {available_markets[0]['minute_file']}")
            print(f"    {Colors.CYAN}{spec_info}{Colors.RESET}")
            print(f"\n{Colors.GREEN}Auto-selecting {market} for training.{Colors.RESET}")
            return market

        # Multiple markets - show selection menu
        print(f"{Colors.GREEN}Detected {len(available_markets)} markets:{Colors.RESET}\n")

        for i, market_info in enumerate(available_markets, 1):
            market = market_info['market']
            market_spec = get_market_spec(market)
            spec_info = f"${market_spec.contract_multiplier} x {market_spec.tick_size} tick = ${market_spec.tick_value:.2f}"

            print(f"{Colors.BOLD}  {i}. {market:<8}{Colors.RESET} - {market_info['minute_file']:<25}")
            print(f"     {Colors.CYAN}{spec_info} | Commission: ${market_spec.commission}/side{Colors.RESET}")
            print()

        print(f"{Colors.YELLOW}  0. Cancel{Colors.RESET}")
        print()

        valid_choices = [str(i) for i in range(len(available_markets) + 1)]
        choice = get_user_input(
            f"{Colors.BOLD}Select market to train on (0-{len(available_markets)}): {Colors.RESET}",
            valid_choices
        )

        if choice == "0" or choice is None:
            print(f"\n{Colors.YELLOW}Market selection cancelled.{Colors.RESET}")
            return None

        selected_market = available_markets[int(choice) - 1]['market']
        print(f"\n{Colors.GREEN}Selected market: {Colors.BOLD}{selected_market}{Colors.RESET}")
        return selected_market

    except Exception as e:
        print(f"\n{Colors.RED}Error detecting markets: {str(e)}{Colors.RESET}")
        return None


def select_hardware_profile(project_dir: Path) -> Optional[str]:
    """
    Prompt user to select a hardware profile.
    
    Args:
        project_dir: Project root directory
        
    Returns:
        Path to selected profile or None
    """
    profile_dir = project_dir / "config" / "hardware_profiles"
    if not profile_dir.exists():
        return None

    profiles = list(profile_dir.glob("*.yaml"))
    if not profiles:
        return None

    print(f"\n{Colors.BOLD}Hardware Optimization:{Colors.RESET}")
    print(f"{Colors.CYAN}Hardware profiles found. Select one to optimize training speed:{Colors.RESET}")
    print(f"{Colors.CYAN}  0. None (Use defaults){Colors.RESET}")
    
    for i, profile in enumerate(profiles, 1):
        print(f"{Colors.CYAN}  {i}. {profile.stem} ({profile.name}){Colors.RESET}")

    choice = get_user_input(
        f"{Colors.YELLOW}Select profile (0-{len(profiles)}): {Colors.RESET}",
        [str(i) for i in range(len(profiles) + 1)]
    )

    if choice is None or choice == "0":
        return None
    
    return str(profiles[int(choice) - 1])
