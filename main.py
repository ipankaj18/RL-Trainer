#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RL TRAINER - Interactive Command-Line Menu System
Comprehensive interface for managing reinforcement learning trading system

Features:
- Requirements installation with progress tracking
- Data processing with instrument selection
- Training model management (test/production)
- Model evaluation with result logging
- Colored output and error handling
- Input validation and logging

Author: RL Trading System Team
Date: October 2025
"""

import os
import sys
import subprocess
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

# Import model utilities
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from model_utils import detect_models_in_folder, display_model_selection

# Try to import colorama for colored output
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    print("Warning: colorama not installed. Install with: pip install colorama")
    COLORAMA_AVAILABLE = False

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    print("Warning: tqdm not installed. Install with: pip install tqdm")
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


class RLTrainerMenu:
    """Main menu system for RL Trading System."""
    
    def __init__(self):
        """Initialize the menu system."""
        self.setup_logging()
        self.project_dir = Path(__file__).resolve().parent
        self.src_dir = self.project_dir / "src"
        self.logs_dir = self.project_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        
        # Valid instruments for data processing
        self.valid_instruments = ["NQ", "ES", "YM", "RTY", "MNQ", "MES", "M2K", "MYM"]
        
        # Menu options
        self.main_menu_options = {
            "1": "Requirements Installation",
            "2": "Data Processing",
            "3": "Training Model",
            "4": "Evaluator",
            "5": "Exit"
        }
        
        self.training_menu_options = {
            "1": "Complete Training Pipeline (Test Mode)",
            "2": "Complete Training Pipeline (Production Mode)",
            "3": "Continue Training from Existing Model",
            "4": "Back to Main Menu"
        }
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Create main log file
        log_file = log_dir / f"rl_trainer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("RL Trainer Menu System Initialized")
    
    def clear_screen(self):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def display_banner(self):
        """Display the RL TRAINER banner with Unicode block characters similar to CLAUDE CODE style."""
        # RL TRAINER ASCII art using full Unicode block characters - large blocky style
        rl_trainer_art = [
            "███████  ██      ",
            "██   ██  ██      ",
            "███████  ██      ",
            "██████   ██      ",
            "██  ███  ██      ",
            "██   ██  ███████ ",
            "",
            "████████ ███████   █████  ██ ███    ██ ███████ ███████ ",
            "   ██    ██   ██  ██   ██ ██ ████   ██ ██      ██   ██ ",
            "   ██    ███████  ███████ ██ ██ ██  ██ █████   ███████ ",
            "   ██    ██████   ██   ██ ██ ██  ██ ██ ██      ██████  ",
            "   ██    ██  ███  ██   ██ ██ ██   ████ ██      ██  ███ ",
            "   ██    ██   ██  ██   ██ ██ ██    ███ ███████ ██   ██ "
        ]
        
        # Alternative ultra-blocky style (more similar to screenshot)
        blocky_style = [
            "██████╗ ██╗     ",
            "██╔══██╗██║     ",
            "██████╔╝██║     ",
            "██╔══██╗██║     ",
            "██║  ██║███████╗",
            "╚═╝  ╚═╝╚══════╝",
            "",
            "████████╗██████╗  █████╗ ██╗███╗   ██╗███████╗██████╗ ",
            "╚══██╔══╝██╔══██╗██╔══██╗██║████╗  ██║██╔════╝██╔══██╗",
            "   ██║   ██████╔╝███████║██║██╔██╗ ██║█████╗  ██████╔╝",
            "   ██║   ██╔══██╗██╔══██║██║██║╚██╗██║██╔══╝  ██╔══██╗",
            "   ██║   ██║  ██║██║  ██║██║██║ ╚████║███████╗██║  ██║",
            "   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝"
        ]

        # Pure block characters style (most similar to CLAUDE CODE screenshot)
        pure_blocks = """
█████████  ██         
██     ██  ██         
█████████  ██         
██     ██  ██         
██     ██  ██         
██     ██  ███████████

████████ ████████   ████████  ████  ██     ██ ████████ ████████  
   ██    ██     ██  ██    ██   ██   ████   ██ ██       ██     ██ 
   ██    ████████   ████████   ██   ██ ██  ██ ████████ ████████  
   ██    ██   ██    ██    ██   ██   ██  ██ ██ ██       ██   ██   
   ██    ██     ██  ██    ██   ██   ██   ████ ████████ ██     ██ """

        # Using the blocky_style which looks cleaner
        banner_color = f"{Colors.CYAN}{Colors.BOLD}"

        header_text = " Welcome to the RL TRAINER research preview! "
        header_border = "═" * (len(header_text) + 2)

        banner_sections = [
            f"{Colors.YELLOW}╔{header_border}╗{Colors.RESET}",
            f"{Colors.YELLOW}║ {Colors.RESET}{Colors.WHITE}{header_text}{Colors.RESET}{Colors.YELLOW} ║{Colors.RESET}",
            f"{Colors.YELLOW}╚{header_border}╝{Colors.RESET}",
            "",
            f"{banner_color}" + "\n".join(blocky_style) + f"{Colors.RESET}",
            "",
            f"{Colors.CYAN}{Colors.BOLD}{'═' * 78}{Colors.RESET}",
            f"{Colors.CYAN}{Colors.BOLD}  Reinforcement Learning Trading System - Interactive Menu Interface{Colors.RESET}",
            f"{Colors.CYAN}{Colors.BOLD}  Version 1.0.0 - October 2025{Colors.RESET}",
            f"{Colors.CYAN}{Colors.BOLD}{'═' * 78}{Colors.RESET}",
            "",
            f"{Colors.GREEN}{Colors.BOLD}Welcome to RL TRAINER! Your comprehensive trading system management tool.{Colors.RESET}",
        ]

        print("\n".join(banner_sections))

    def detect_and_select_market(self) -> Optional[str]:
        """
        Detect available market data files and prompt user to select one.

        Returns:
            Selected market symbol (e.g., 'ES', 'NQ') or None if cancelled
        """
        try:
            # Import market detection utilities from model_utils
            # Note: src directory should already be in sys.path from line 29
            from model_utils import detect_available_markets
            from market_specs import get_market_spec

            # Detect available markets
            data_dir = self.project_dir / 'data'
            available_markets = detect_available_markets(str(data_dir))

            if not available_markets:
                print(f"\n{Colors.RED}No market data files found in data/ directory.{Colors.RESET}")
                print(f"{Colors.YELLOW}Please run 'Data Processing' first to prepare market data.{Colors.RESET}")
                return None

            # Display header
            print(f"\n{Colors.BOLD}{Colors.CYAN}╔══════════════════════════════════════════════════════════════╗{Colors.RESET}")
            print(f"{Colors.BOLD}{Colors.CYAN}║                   MARKET SELECTION                            ║{Colors.RESET}")
            print(f"{Colors.BOLD}{Colors.CYAN}╚══════════════════════════════════════════════════════════════╝{Colors.RESET}")
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

            # Get user selection
            valid_choices = [str(i) for i in range(len(available_markets) + 1)]
            choice = self.get_user_input(
                f"{Colors.BOLD}Select market to train on (0-{len(available_markets)}): {Colors.RESET}",
                valid_choices
            )

            if choice == "0" or choice is None:
                print(f"\n{Colors.YELLOW}Market selection cancelled.{Colors.RESET}")
                return None

            # Return selected market symbol
            selected_market = available_markets[int(choice) - 1]['market']
            print(f"\n{Colors.GREEN}Selected market: {Colors.BOLD}{selected_market}{Colors.RESET}")
            return selected_market

        except Exception as e:
            self.logger.error(f"Error detecting markets: {str(e)}", exc_info=True)
            print(f"\n{Colors.RED}Error detecting markets: {str(e)}{Colors.RESET}")
            return None

    def display_main_menu(self):
        """Display the main menu options."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}╔══════════════════════════════════════════════════════════════╗{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}║                        MAIN MENU                              ║{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}╚══════════════════════════════════════════════════════════════╝{Colors.RESET}")
        print()
        
        for key, value in self.main_menu_options.items():
            if key == "5":  # Exit option
                print(f"{Colors.RED}  {key}. {value}{Colors.RESET}")
            else:
                print(f"{Colors.GREEN}  {key}. {value}{Colors.RESET}")
        
        print()
    
    def display_training_menu(self):
        """Display the training submenu options."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}╔══════════════════════════════════════════════════════════════╗{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}║                      TRAINING MENU                            ║{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}╚══════════════════════════════════════════════════════════════╝{Colors.RESET}")
        print()
        for key, value in self.training_menu_options.items():
            color = Colors.YELLOW if key == "4" else Colors.GREEN
            print(f"{color}  {key}. {value}{Colors.RESET}")
        print()

    
    def get_user_input(self, prompt: str, valid_options: Optional[List[str]] = None) -> Optional[str]:
        """
        Get and validate user input.
        
        Args:
            prompt: The prompt to display to the user
            valid_options: List of valid input options
            
        Returns:
            Validated user input
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
    
    def validate_instrument(self, instrument: str) -> bool:
        """
        Validate instrument selection.
        
        Args:
            instrument: The instrument to validate
            
        Returns:
            True if valid, False otherwise
        """
        return instrument.upper() in self.valid_instruments
    
    def display_instruments(self):
        """Display available instruments for selection."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}Available Instruments:{Colors.RESET}")
        for i, instrument in enumerate(self.valid_instruments, 1):
            print(f"{Colors.GREEN}  {i}. {instrument}{Colors.RESET}")
        print()
    
    def select_instrument(self, prompt_message: str = "Enter instrument number or name") -> Optional[str]:
        """
        Display instruments and get user selection.
        
        Args:
            prompt_message: Custom prompt for the selection
            
        Returns:
            Selected instrument (e.g., 'NQ', 'ES') or None if cancelled
        """
        self.display_instruments()
        
        while True:
            instrument_input = self.get_user_input(
                f"{Colors.YELLOW}{prompt_message} (e.g., '1' or 'ES'): {Colors.RESET}"
            )
            
            if instrument_input is None:
                return None
                
            if instrument_input.lower() in ["cancel", "exit"]:
                return None
            
            # Check if input is a number
            if instrument_input.isdigit():
                instrument_idx = int(instrument_input) - 1
                if 0 <= instrument_idx < len(self.valid_instruments):
                    return self.valid_instruments[instrument_idx]
                else:
                    print(f"{Colors.RED}Invalid number. Please choose 1-{len(self.valid_instruments)}.{Colors.RESET}")
            else:
                # Check if input is a valid instrument name
                if self.validate_instrument(instrument_input):
                    return instrument_input.upper()
                else:
                    print(f"{Colors.RED}Invalid instrument. Please choose from the list.{Colors.RESET}")
    
    def run_command_with_progress(self, command: List[str], description: str, 
                              log_file: str = None) -> Tuple[bool, str]:
        """
        Run a command with progress indication and logging.
        
        Args:
            command: Command to execute as list of strings
            description: Description of the operation
            log_file: Log file to save output
            
        Returns:
            Tuple of (success, output)
        """
        print(f"\n{Colors.YELLOW}{description}{Colors.RESET}")
        print(f"{Colors.CYAN}Executing: {' '.join(command)}{Colors.RESET}")
        print(f"{Colors.MAGENTA}{'=' * 60}{Colors.RESET}")

        try:
            # Always run from project root directory
            working_directory = self.project_dir

            # Set up environment with project root in PYTHONPATH for imports
            env = os.environ.copy()
            pythonpath = str(self.project_dir)
            if 'PYTHONPATH' in env:
                env['PYTHONPATH'] = f"{pythonpath}{os.pathsep}{env['PYTHONPATH']}"
            else:
                env['PYTHONPATH'] = pythonpath

            # Start the process
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,  # Prevent interactive prompts from hanging
                encoding='utf-8',  # Explicitly use UTF-8 to handle Unicode characters (✅, ⚠️, etc.)
                errors='replace',  # Replace invalid characters instead of crashing
                cwd=working_directory,
                env=env
            )

            output_lines = []

            # Read output line by line
            if TQDM_AVAILABLE:
                # Use tqdm for progress indication
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
                # Simple output without progress bar
                output, _ = process.communicate()
                output_lines = output.split('\n') if output else []
                for line in output_lines:
                    cleaned = line.strip()
                    if cleaned:
                        print(f"{Colors.WHITE}{cleaned}{Colors.RESET}")
            
            # Check return code
            success = process.returncode == 0
            
            if success:
                print(f"\n{Colors.GREEN}✓ {description} completed successfully!{Colors.RESET}")
            else:
                print(f"\n{Colors.RED}✗ {description} failed with return code {process.returncode}{Colors.RESET}")
            
            # Save to log file if specified
            if log_file:
                log_path = self.logs_dir / log_file
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
    
    def check_package_installed(self, package_name: str) -> bool:
        """
        Check if a package is installed.

        Args:
            package_name: Name of the package to check

        Returns:
            True if installed, False otherwise
        """
        try:
            # Handle package names with version specifiers
            pkg = package_name.split('>=')[0].split('==')[0].split('<')[0].strip()
            
            # Map package names to their import names
            # (requirements.txt name -> Python import name)
            package_name_mapping = {
                'pyyaml': 'yaml',
                'ipython': 'IPython',
                'stable-baselines3': 'stable_baselines3',
                'sb3-contrib': 'sb3_contrib',
                'scikit-learn': 'sklearn',
                'pillow': 'PIL',
                'opencv-python': 'cv2',
                'python-dateutil': 'dateutil',
                'attrs': 'attr',
            }
            
            # Use mapped name if available, otherwise use the package name with hyphens replaced by underscores
            import_name = package_name_mapping.get(pkg.lower(), pkg.replace('-', '_'))
            
            __import__(import_name)
            return True
        except ImportError:
            return False

    def check_installed_requirements(self) -> Tuple[List[str], List[str]]:
        """
        Check which requirements are already installed.

        Returns:
            Tuple of (installed_packages, missing_packages)
        """
        requirements_file = self.project_dir / "requirements.txt"
        installed = []
        missing = []

        with open(requirements_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue

                # Extract package name
                package = line.split('>=')[0].split('==')[0].split('<')[0].strip()

                if self.check_package_installed(package):
                    installed.append(package)
                else:
                    missing.append(package)

        return installed, missing

    def install_requirements(self):
        """Install requirements using pip with smart checking."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}╔══════════════════════════════════════════════════════════════╗{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}║                  REQUIREMENTS INSTALLATION                    ║{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}╚══════════════════════════════════════════════════════════════╝{Colors.RESET}")

        # Check if requirements.txt exists
        requirements_file = self.project_dir / "requirements.txt"
        if not requirements_file.exists():
            print(f"{Colors.RED}Error: requirements.txt not found in project directory{Colors.RESET}")
            return False

        print(f"{Colors.GREEN}Found requirements.txt at: {requirements_file}{Colors.RESET}")

        # Check what's already installed
        print(f"\n{Colors.CYAN}Checking installed packages...{Colors.RESET}")
        installed, missing = self.check_installed_requirements()

        total = len(installed) + len(missing)

        # Display status
        print(f"\n{Colors.BOLD}Installation Status:{Colors.RESET}")
        print(f"{Colors.GREEN}  ✓ Installed: {len(installed)}/{total} packages{Colors.RESET}")
        if missing:
            print(f"{Colors.YELLOW}  ✗ Missing: {len(missing)}/{total} packages{Colors.RESET}")

        # If all installed
        if not missing:
            print(f"\n{Colors.GREEN}{Colors.BOLD}✓ All requirements are already installed!{Colors.RESET}")
            print(f"\n{Colors.CYAN}Installed packages:{Colors.RESET}")
            for pkg in installed[:8]:  # Show first 8
                print(f"  • {pkg}")
            if len(installed) > 8:
                print(f"  ... and {len(installed) - 8} more")

            print(f"\n{Colors.YELLOW}Options:{Colors.RESET}")
            print(f"  {Colors.GREEN}1. Return to Main Menu{Colors.RESET}")
            print(f"  {Colors.CYAN}2. Reinstall All Packages (if having issues){Colors.RESET}")
            print(f"  {Colors.CYAN}3. Upgrade All Packages{Colors.RESET}")

            choice = self.get_user_input(
                f"\n{Colors.YELLOW}Select option (1-3): {Colors.RESET}",
                ["1", "2", "3"]
            )

            if choice is None or choice == "1":
                print(f"{Colors.GREEN}Returning to main menu...{Colors.RESET}")
                return True
            elif choice == "2":
                command = [sys.executable, "-m", "pip", "install", "--force-reinstall", "-r", "requirements.txt"]
                description = "Reinstalling All Requirements"
            else:  # choice == "3"
                command = [sys.executable, "-m", "pip", "install", "--upgrade", "-r", "requirements.txt"]
                description = "Upgrading All Requirements"

        # If some missing
        else:
            print(f"\n{Colors.YELLOW}Missing packages:{Colors.RESET}")
            for pkg in missing:
                print(f"  • {pkg}")

            if installed:
                print(f"\n{Colors.GREEN}Already installed:{Colors.RESET}")
                for pkg in installed[:5]:  # Show first 5
                    print(f"  • {pkg}")
                if len(installed) > 5:
                    print(f"  ... and {len(installed) - 5} more")

            print(f"\n{Colors.YELLOW}Options:{Colors.RESET}")
            print(f"  {Colors.GREEN}1. Install Missing Packages Only{Colors.RESET}")
            print(f"  {Colors.CYAN}2. Reinstall All Packages{Colors.RESET}")
            print(f"  {Colors.RED}3. Cancel / Return to Main Menu{Colors.RESET}")

            choice = self.get_user_input(
                f"\n{Colors.YELLOW}Select option (1-3): {Colors.RESET}",
                ["1", "2", "3"]
            )

            if choice is None or choice == "3":
                print(f"{Colors.YELLOW}Installation cancelled. Returning to main menu...{Colors.RESET}")
                return False
            elif choice == "1":
                command = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
                description = "Installing Missing Requirements"
            else:  # choice == "2"
                command = [sys.executable, "-m", "pip", "install", "--force-reinstall", "-r", "requirements.txt"]
                description = "Reinstalling All Requirements"

        # Run installation
        print()  # Blank line for readability
        success, output = self.run_command_with_progress(
            command,
            description,
            "installation.log"
        )

        if success:
            print(f"\n{Colors.GREEN}✓ Installation completed successfully!{Colors.RESET}")
            print(f"{Colors.CYAN}All dependencies are now installed.{Colors.RESET}")
            print(f"{Colors.CYAN}You can now use all features of the RL Trading System.{Colors.RESET}")
        else:
            print(f"\n{Colors.RED}✗ Installation failed. Check logs for details.{Colors.RESET}")
            print(f"{Colors.YELLOW}Tip: Try running 'pip install -r requirements.txt' manually{Colors.RESET}")

        return success
    
    def process_data(self):
        """Handle data processing with submenu for full or incremental update."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}╔══════════════════════════════════════════════════════════════╗{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}║                     DATA PROCESSING                            ║{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}╚══════════════════════════════════════════════════════════════╝{Colors.RESET}")

        # Show data processing options
        print(f"\n{Colors.BOLD}Choose processing mode:{Colors.RESET}")
        print(f"\n{Colors.GREEN}  1. Process New Data (Full){Colors.RESET}")
        print(f"     {Colors.CYAN}• Complete data processing from .zip files{Colors.RESET}")
        print(f"     {Colors.CYAN}• Overwrites existing data{Colors.RESET}")
        print(f"     {Colors.CYAN}• Use for first-time setup or full reprocessing{Colors.RESET}")

        print(f"\n{Colors.GREEN}  2. Update Existing Data (Incremental){Colors.RESET}")
        print(f"     {Colors.CYAN}• Adds only NEW data from .zip files{Colors.RESET}")
        print(f"     {Colors.CYAN}• Keeps existing data + appends new{Colors.RESET}")
        print(f"     {Colors.CYAN}• 10x faster for small updates{Colors.RESET}")
        print(f"     {Colors.CYAN}• Creates automatic backups{Colors.RESET}")

        print(f"\n{Colors.YELLOW}  3. Back to Main Menu{Colors.RESET}")

        choice = self.get_user_input(
            f"\n{Colors.YELLOW}Select option (1-3): {Colors.RESET}",
            ["1", "2", "3"]
        )

        if choice is None or choice == "3":
            print(f"{Colors.YELLOW}Returning to main menu...{Colors.RESET}")
            return False
        elif choice == "1":
            return self.process_data_full()
        elif choice == "2":
            return self.process_data_incremental()

    def process_data_full(self):
        """Handle full data processing (original method)."""
        print(f"\n{Colors.BOLD}{Colors.GREEN}PROCESS NEW DATA (FULL){Colors.RESET}")
        print(f"{Colors.YELLOW}This will process data from scratch and overwrite existing files.{Colors.RESET}\n")

        # Get instrument selection using helper method
        instrument = self.select_instrument()
        if instrument is None:
            print(f"{Colors.YELLOW}Data processing cancelled.{Colors.RESET}")
            return False

        print(f"{Colors.GREEN}Selected instrument: {instrument}{Colors.RESET}")

        # Confirm selection
        confirm = self.get_user_input(
            f"{Colors.YELLOW}Process data for {instrument}? (y/n): {Colors.RESET}",
            ["y", "n", "Y", "N"]
        )

        if confirm is None or confirm.lower() != 'y':
            print(f"{Colors.YELLOW}Data processing cancelled.{Colors.RESET}")
            return False

        # Run data processing
        update_script = self.src_dir / "update_training_data.py"
        if not update_script.exists():
            print(f"{Colors.RED}Error: update_training_data.py not found{Colors.RESET}")
            return False

        command = [sys.executable, str(update_script), "--market", instrument]
        success, output = self.run_command_with_progress(
            command,
            f"Processing {instrument} Data",
            f"data_processing_{instrument.lower()}.log"
        )

        if success:
            print(f"\n{Colors.GREEN}✓ Data processing completed for {instrument}!{Colors.RESET}")
            print(f"{Colors.CYAN}Data files created: {instrument}_D1M.csv, {instrument}_D1S.csv{Colors.RESET}")
        else:
            print(f"\n{Colors.RED}✗ Data processing failed for {instrument}. Check logs for details.{Colors.RESET}")

        return success

    def process_data_incremental(self):
        """Handle incremental data update (adds new data to existing)."""
        print(f"\n{Colors.BOLD}{Colors.GREEN}UPDATE EXISTING DATA (INCREMENTAL){Colors.RESET}")
        print(f"{Colors.CYAN}This will detect new data and append it to existing files.{Colors.RESET}\n")

        # Get instrument selection using helper method
        instrument = self.select_instrument()
        if instrument is None:
            print(f"{Colors.YELLOW}Incremental update cancelled.{Colors.RESET}")
            return False

        print(f"{Colors.GREEN}Selected instrument: {instrument}{Colors.RESET}")

        # Run incremental update
        incremental_script = self.src_dir / "incremental_data_updater.py"
        if not incremental_script.exists():
            print(f"{Colors.RED}Error: incremental_data_updater.py not found{Colors.RESET}")
            return False

        print(f"\n{Colors.CYAN}Running incremental update for {instrument}...{Colors.RESET}")
        print(f"{Colors.YELLOW}Note: You will be asked to confirm before any changes are made.{Colors.RESET}\n")

        # Set up environment with project root in PYTHONPATH for imports
        env = os.environ.copy()
        pythonpath = str(self.project_dir)
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = f"{pythonpath}{os.pathsep}{env['PYTHONPATH']}"
        else:
            env['PYTHONPATH'] = pythonpath

        # Run with subprocess to show interactive output
        command = [sys.executable, str(incremental_script), "--market", instrument]

        try:
            # Run interactively (user will see prompts and can respond)
            result = subprocess.run(
                command,
                cwd=self.project_dir,
                env=env,  # Pass environment with PYTHONPATH configured
                check=False
            )

            success = (result.returncode == 0)

            if success:
                print(f"\n{Colors.GREEN}✓ Incremental update completed for {instrument}!{Colors.RESET}")
                print(f"{Colors.CYAN}Existing data has been updated with new dates.{Colors.RESET}")
            else:
                print(f"\n{Colors.YELLOW}Incremental update exited with code {result.returncode}{Colors.RESET}")

            return success

        except Exception as e:
            print(f"\n{Colors.RED}✗ Incremental update failed: {str(e)}{Colors.RESET}")
            self.logger.error(f"Incremental update error: {str(e)}")
            return False
    
    def train_model(self):
        """Handle model training with test/production options."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}╔══════════════════════════════════════════════════════════════╗{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}║                      TRAINING MODEL                             ║{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}╚══════════════════════════════════════════════════════════════╝{Colors.RESET}")

        while True:
            self.display_training_menu()
            choice = self.get_user_input(
                f"{Colors.YELLOW}Select training option: {Colors.RESET}",
                list(self.training_menu_options.keys())
            )

            if choice is None:
                print(f"{Colors.YELLOW}Training menu cancelled. Returning to main menu...{Colors.RESET}")
                break

            if choice == "1":
                self.run_complete_pipeline_test()
                break
            elif choice == "2":
                self.run_complete_pipeline_production()
                break
            elif choice == "3":
                self.continue_training_from_model()
                break
            elif choice == "4":
                print(f"{Colors.YELLOW}Returning to main menu...{Colors.RESET}")
                break
            else:
                print(f"{Colors.RED}Invalid option. Please try again.{Colors.RESET}")

    def run_complete_pipeline_test(self):
        """Run complete 3-phase training pipeline in test mode."""
        print(f"\n{Colors.BOLD}{Colors.YELLOW}COMPLETE TRAINING PIPELINE - TEST MODE{Colors.RESET}")
        print(f"{Colors.CYAN}This will run all 3 phases sequentially with reduced timesteps.{Colors.RESET}")
        print(f"\n{Colors.BOLD}Pipeline Overview:{Colors.RESET}")
        print(f"{Colors.GREEN}  Phase 1: Entry Learning (5-10 minutes){Colors.RESET}")
        print(f"{Colors.GREEN}  Phase 2: Position Management (10-15 minutes){Colors.RESET}")
        print(f"{Colors.GREEN}  Phase 3: Extended RL (261D Observations) (15-20 minutes){Colors.RESET}")
        print(f"{Colors.YELLOW}  Total Estimated Time: 30-45 minutes{Colors.RESET}")
        print()

        # Confirm
        confirm = self.get_user_input(
            f"{Colors.YELLOW}Proceed with complete pipeline test? (y/n): {Colors.RESET}",
            ["y", "n", "Y", "N"]
        )

        if confirm is None or confirm.lower() != 'y':
            print(f"{Colors.YELLOW}Pipeline test cancelled.{Colors.RESET}")
            return False

        # Market selection - ONCE for entire pipeline
        selected_market = self.detect_and_select_market()
        if selected_market is None:
            print(f"{Colors.YELLOW}Pipeline cancelled - no market selected.{Colors.RESET}")
            return False

        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}Starting Complete Training Pipeline for {selected_market}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.RESET}\n")

        import time
        pipeline_start_time = time.time()

        # ============================================
        # PHASE 1: Entry Learning
        # ============================================
        phase1_script = self.src_dir / "train_phase1.py"
        if not phase1_script.exists():
            print(f"{Colors.RED}Error: Phase 1 training script not found{Colors.RESET}")
            return False

        print(f"\n{Colors.BOLD}{Colors.GREEN}[1/3] PHASE 1: Entry Learning{Colors.RESET}")
        print(f"{Colors.CYAN}Training agent to recognize entry signals...{Colors.RESET}\n")

        phase1_start = time.time()
        command = [
            sys.executable, str(phase1_script),
            "--test",
            "--market", selected_market,
            "--non-interactive"
        ]

        success, output = self.run_command_with_progress(
            command,
            "Phase 1: Entry Learning (Test Mode)",
            "pipeline_test_phase1.log"
        )

        phase1_duration = (time.time() - phase1_start) / 60

        if not success:
            print(f"\n{Colors.RED}✗ Phase 1 failed. Pipeline stopped.{Colors.RESET}")
            print(f"{Colors.YELLOW}Check log: logs/pipeline_test_phase1.log{Colors.RESET}")
            return False

        # Sanity check: even in test mode, ensure evaluation artifacts exist
        try:
            from pipeline.phase_guard import PhaseGuard

            eval_path = PhaseGuard._find_newest_eval_file("logs/phase1")
            if not Path(eval_path).exists():
                print(f"\n{Colors.RED}✗ Phase 1 missing evaluation results at {eval_path}{Colors.RESET}")
                print(f"{Colors.YELLOW}Pipeline stopped early so production guard would not be surprised.{Colors.RESET}")
                return False
        except Exception as exc:
            print(f"\n{Colors.RED}✗ Failed to verify Phase 1 evaluation artifacts: {exc}{Colors.RESET}")
            return False

        print(f"{Colors.GREEN}✓ Phase 1 completed in {phase1_duration:.1f} minutes{Colors.RESET}")

        # ============================================
        # PHASE 2: Position Management
        # ============================================
        phase2_script = self.src_dir / "train_phase2.py"
        if not phase2_script.exists():
            print(f"{Colors.RED}Error: Phase 2 training script not found{Colors.RESET}")
            return False

        print(f"\n{Colors.BOLD}{Colors.GREEN}[2/3] PHASE 2: Position Management{Colors.RESET}")
        print(f"{Colors.CYAN}Training agent for position and risk management...{Colors.RESET}\n")

        phase2_start = time.time()
        command = [
            sys.executable, str(phase2_script),
            "--test",
            "--market", selected_market,
            "--non-interactive"
        ]

        success, output = self.run_command_with_progress(
            command,
            "Phase 2: Position Management (Test Mode)",
            "pipeline_test_phase2.log"
        )

        phase2_duration = (time.time() - phase2_start) / 60

        if not success:
            print(f"\n{Colors.RED}✗ Phase 2 failed. Pipeline stopped.{Colors.RESET}")
            print(f"{Colors.YELLOW}Check log: logs/pipeline_test_phase2.log{Colors.RESET}")
            return False

        # Sanity check: ensure Phase 2 evaluation exists in test mode as well
        try:
            from pipeline.phase_guard import PhaseGuard

            eval_path = PhaseGuard._find_newest_eval_file("logs/phase2")
            if not Path(eval_path).exists():
                print(f"\n{Colors.RED}✗ Phase 2 missing evaluation results at {eval_path}{Colors.RESET}")
                print(f"{Colors.YELLOW}Pipeline stopped to surface the missing eval artifact before production.{Colors.RESET}")
                return False
        except Exception as exc:
            print(f"\n{Colors.RED}✗ Failed to verify Phase 2 evaluation artifacts: {exc}{Colors.RESET}")
            return False

        print(f"{Colors.GREEN}✓ Phase 2 completed in {phase2_duration:.1f} minutes{Colors.RESET}")

        # ============================================
        # PHASE 3: Hybrid LLM Agent
        # ============================================

        # Check GPU before Phase 3
        print(f"\n{Colors.BOLD}{Colors.GREEN}[3/3] PHASE 3: Hybrid LLM Agent{Colors.RESET}")
        print(f"{Colors.CYAN}Checking GPU availability...{Colors.RESET}")

        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"{Colors.GREEN}✓ GPU detected: {gpu_name} ({gpu_mem:.1f}GB){Colors.RESET}")

                if gpu_mem < 8:
                    print(f"{Colors.YELLOW}⚠ Warning: GPU memory may be insufficient for optimal LLM performance.{Colors.RESET}")
                    proceed = self.get_user_input(
                        f"{Colors.YELLOW}Continue with Phase 3? (y/n): {Colors.RESET}",
                        ["y", "n", "Y", "N"]
                    )
                    if proceed is None or proceed.lower() != 'y':
                        print(f"{Colors.YELLOW}Phase 3 skipped. Pipeline completed through Phase 2.{Colors.RESET}")
                        return True
            else:
                print(f"{Colors.YELLOW}⚠ No GPU detected. LLM inference will be slow.{Colors.RESET}")
                proceed = self.get_user_input(
                    f"{Colors.YELLOW}Continue with CPU (slow)? (y/n): {Colors.RESET}",
                    ["y", "n", "Y", "N"]
                )
                if proceed is None or proceed.lower() != 'y':
                    print(f"{Colors.YELLOW}Phase 3 skipped. Pipeline completed through Phase 2.{Colors.RESET}")
                    return True
        except ImportError:
            print(f"{Colors.YELLOW}⚠ PyTorch not available. Phase 3 requires PyTorch.{Colors.RESET}")
            print(f"{Colors.YELLOW}Pipeline completed through Phase 2 only.{Colors.RESET}")
            return True

        phase3_script = self.src_dir / "train_phase3_llm.py"
        if not phase3_script.exists():
            print(f"{Colors.RED}Error: Phase 3 training script not found{Colors.RESET}")
            print(f"{Colors.YELLOW}Pipeline completed through Phase 2 only.{Colors.RESET}")
            return True  # Not a failure, just Phase 3 not available

        print(f"\n{Colors.CYAN}Training hybrid RL + LLM agent...{Colors.RESET}")
        print(f"{Colors.CYAN}Note: Phase 3 uses 261D observations (no LLM - pure RL mode){Colors.RESET}\n")

        phase3_start = time.time()
        command = [
            sys.executable, str(phase3_script),
            "--test",
            "--market", selected_market,
            "--non-interactive"
        ]

        success, output = self.run_command_with_progress(
            command,
            "Phase 3: Extended RL - 261D Obs (Test Mode)",
            "pipeline_test_phase3.log"
        )

        phase3_duration = (time.time() - phase3_start) / 60

        if not success:
            print(f"\n{Colors.RED}✗ Phase 3 failed.{Colors.RESET}")
            print(f"{Colors.YELLOW}Check log: logs/pipeline_test_phase3.log{Colors.RESET}")
            print(f"{Colors.CYAN}Note: Phase 1 and 2 models were successfully trained.{Colors.RESET}")
            return False

        print(f"{Colors.GREEN}✓ Phase 3 completed in {phase3_duration:.1f} minutes{Colors.RESET}")

        # ============================================
        # Pipeline Complete
        # ============================================
        total_duration = (time.time() - pipeline_start_time) / 60

        print(f"\n{Colors.BOLD}{Colors.GREEN}{'='*60}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.GREEN}✓ COMPLETE PIPELINE FINISHED SUCCESSFULLY{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.GREEN}{'='*60}{Colors.RESET}")
        print(f"\n{Colors.CYAN}Pipeline Summary:{Colors.RESET}")
        print(f"{Colors.WHITE}  Market: {selected_market}{Colors.RESET}")
        print(f"{Colors.WHITE}  Phase 1 Duration: {phase1_duration:.1f} minutes{Colors.RESET}")
        print(f"{Colors.WHITE}  Phase 2 Duration: {phase2_duration:.1f} minutes{Colors.RESET}")
        print(f"{Colors.WHITE}  Phase 3 Duration: {phase3_duration:.1f} minutes{Colors.RESET}")
        print(f"{Colors.YELLOW}  Total Pipeline Time: {total_duration:.1f} minutes{Colors.RESET}")
        print(f"\n{Colors.GREEN}All trained models saved in models/ directory{Colors.RESET}")
        print(f"{Colors.CYAN}You can now run the Evaluator to assess model performance.{Colors.RESET}\n")

        return True

    def run_complete_pipeline_production(self):
        """Run complete 3-phase training pipeline in production mode."""
        print(f"\n{Colors.BOLD}{Colors.YELLOW}COMPLETE TRAINING PIPELINE - PRODUCTION MODE{Colors.RESET}")
        print(f"{Colors.CYAN}This will run full production training on the complete dataset.{Colors.RESET}")
        print(f"{Colors.RED}⚠ WARNING: This is a LONG process!{Colors.RESET}")
        print(f"\n{Colors.BOLD}Pipeline Overview:{Colors.RESET}")
        print(f"{Colors.GREEN}  Phase 1: Entry Learning (6-8 hours){Colors.RESET}")
        print(f"{Colors.GREEN}  Phase 2: Position Management (8-10 hours){Colors.RESET}")
        print(f"{Colors.GREEN}  Phase 3: Extended RL (261D Observations) (12-16 hours){Colors.RESET}")
        print(f"{Colors.YELLOW}  Total Estimated Time: 26-34 hours{Colors.RESET}")
        print(f"\n{Colors.RED}Ensure your system will remain on and connected!{Colors.RESET}")
        print()

        # Confirm
        confirm = self.get_user_input(
            f"{Colors.YELLOW}Proceed with complete production pipeline? (y/n): {Colors.RESET}",
            ["y", "n", "Y", "N"]
        )

        if confirm is None or confirm.lower() != 'y':
            print(f"{Colors.YELLOW}Production pipeline cancelled.{Colors.RESET}")
            return False

        # Market selection - ONCE for entire pipeline
        selected_market = self.detect_and_select_market()
        if selected_market is None:
            print(f"{Colors.YELLOW}Pipeline cancelled - no market selected.{Colors.RESET}")
            return False

        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}Starting Complete Production Pipeline for {selected_market}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.RESET}\n")

        import time
        pipeline_start_time = time.time()

        # ============================================
        # PHASE 1: Entry Learning
        # ============================================
        phase1_script = self.src_dir / "train_phase1.py"
        if not phase1_script.exists():
            print(f"{Colors.RED}Error: Phase 1 training script not found{Colors.RESET}")
            return False

        print(f"\n{Colors.BOLD}{Colors.GREEN}[1/3] PHASE 1: Entry Learning (Production){Colors.RESET}")
        print(f"{Colors.CYAN}Training agent to recognize entry signals (2M timesteps)...{Colors.RESET}\n")

        phase1_start = time.time()
        command = [
            sys.executable, str(phase1_script),
            "--market", selected_market,
            "--non-interactive"
        ]

        success, output = self.run_command_with_progress(
            command,
            "Phase 1: Entry Learning (Production Mode)",
            "pipeline_production_phase1.log"
        )

        phase1_duration = (time.time() - phase1_start) / 3600  # Convert to hours

        if not success:
            print(f"\n{Colors.RED}✗ Phase 1 failed. Pipeline stopped.{Colors.RESET}")
            print(f"{Colors.YELLOW}Check log: logs/pipeline_production_phase1.log{Colors.RESET}")
            return False

        print(f"{Colors.GREEN}✓ Phase 1 completed in {phase1_duration:.1f} hours{Colors.RESET}")

        # ============================================
        # PHASE 2: Position Management
        # ============================================
        phase2_script = self.src_dir / "train_phase2.py"
        if not phase2_script.exists():
            print(f"{Colors.RED}Error: Phase 2 training script not found{Colors.RESET}")
            return False

        print(f"\n{Colors.BOLD}{Colors.GREEN}[2/3] PHASE 2: Position Management (Production){Colors.RESET}")
        print(f"{Colors.CYAN}Training agent for position and risk management (5M timesteps)...{Colors.RESET}\n")

        phase2_start = time.time()
        command = [
            sys.executable, str(phase2_script),
            "--market", selected_market,
            "--non-interactive"
        ]

        success, output = self.run_command_with_progress(
            command,
            "Phase 2: Position Management (Production Mode)",
            "pipeline_production_phase2.log"
        )

        phase2_duration = (time.time() - phase2_start) / 3600  # Convert to hours

        if not success:
            print(f"\n{Colors.RED}✗ Phase 2 failed. Pipeline stopped.{Colors.RESET}")
            print(f"{Colors.YELLOW}Check log: logs/pipeline_production_phase2.log{Colors.RESET}")
            return False

        print(f"{Colors.GREEN}✓ Phase 2 completed in {phase2_duration:.1f} hours{Colors.RESET}")

        # ============================================
        # PHASE 3: Hybrid LLM Agent
        # ============================================

        # Check GPU before Phase 3
        print(f"\n{Colors.BOLD}{Colors.GREEN}[3/3] PHASE 3: Hybrid LLM Agent (Production){Colors.RESET}")
        print(f"{Colors.CYAN}Checking GPU availability...{Colors.RESET}")

        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"{Colors.GREEN}✓ GPU detected: {gpu_name} ({gpu_mem:.1f}GB){Colors.RESET}")

                if gpu_mem < 8:
                    print(f"{Colors.YELLOW}⚠ Warning: GPU memory ({gpu_mem:.1f}GB) is below recommended 8GB.{Colors.RESET}")
                    print(f"{Colors.YELLOW}   LLM training may be slow or fail due to memory constraints.{Colors.RESET}")
                    proceed = self.get_user_input(
                        f"{Colors.YELLOW}Continue with Phase 3? (y/n): {Colors.RESET}",
                        ["y", "n", "Y", "N"]
                    )
                    if proceed is None or proceed.lower() != 'y':
                        print(f"{Colors.YELLOW}Phase 3 skipped. Pipeline completed through Phase 2.{Colors.RESET}")
                        return True
            else:
                print(f"{Colors.RED}✗ No GPU detected. Phase 3 requires GPU for production training.{Colors.RESET}")
                proceed = self.get_user_input(
                    f"{Colors.YELLOW}Attempt CPU training (NOT recommended, very slow)? (y/n): {Colors.RESET}",
                    ["y", "n", "Y", "N"]
                )
                if proceed is None or proceed.lower() != 'y':
                    print(f"{Colors.YELLOW}Phase 3 skipped. Pipeline completed through Phase 2.{Colors.RESET}")
                    return True
        except ImportError:
            print(f"{Colors.RED}✗ PyTorch not available. Phase 3 cannot run.{Colors.RESET}")
            print(f"{Colors.YELLOW}Pipeline completed through Phase 2 only.{Colors.RESET}")
            return True

        phase3_script = self.src_dir / "train_phase3_llm.py"
        if not phase3_script.exists():
            print(f"{Colors.RED}Error: Phase 3 training script not found{Colors.RESET}")
            print(f"{Colors.YELLOW}Pipeline completed through Phase 2 only.{Colors.RESET}")
            return True  # Not a failure, just Phase 3 not available

        print(f"\n{Colors.CYAN}Training hybrid RL + LLM agent (5M timesteps)...{Colors.RESET}")
        print(f"{Colors.CYAN}Note: Phase 3 uses 261D observations (no LLM - pure RL mode){Colors.RESET}\n")

        phase3_start = time.time()
        command = [
            sys.executable, str(phase3_script),
            "--market", selected_market,
            "--non-interactive"
        ]

        success, output = self.run_command_with_progress(
            command,
            "Phase 3: Extended RL - 261D Obs (Production Mode)",
            "pipeline_production_phase3.log"
        )

        phase3_duration = (time.time() - phase3_start) / 3600  # Convert to hours

        if not success:
            print(f"\n{Colors.RED}✗ Phase 3 failed.{Colors.RESET}")
            print(f"{Colors.YELLOW}Check log: logs/pipeline_production_phase3.log{Colors.RESET}")
            print(f"{Colors.CYAN}Note: Phase 1 and 2 models were successfully trained.{Colors.RESET}")
            return False

        print(f"{Colors.GREEN}✓ Phase 3 completed in {phase3_duration:.1f} hours{Colors.RESET}")

        # ============================================
        # Pipeline Complete
        # ============================================
        total_duration = (time.time() - pipeline_start_time) / 3600  # Convert to hours

        print(f"\n{Colors.BOLD}{Colors.GREEN}{'='*60}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.GREEN}✓ COMPLETE PRODUCTION PIPELINE FINISHED SUCCESSFULLY{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.GREEN}{'='*60}{Colors.RESET}")
        print(f"\n{Colors.CYAN}Pipeline Summary:{Colors.RESET}")
        print(f"{Colors.WHITE}  Market: {selected_market}{Colors.RESET}")
        print(f"{Colors.WHITE}  Phase 1 Duration: {phase1_duration:.1f} hours{Colors.RESET}")
        print(f"{Colors.WHITE}  Phase 2 Duration: {phase2_duration:.1f} hours{Colors.RESET}")
        print(f"{Colors.WHITE}  Phase 3 Duration: {phase3_duration:.1f} hours{Colors.RESET}")
        print(f"{Colors.YELLOW}  Total Pipeline Time: {total_duration:.1f} hours ({total_duration/24:.1f} days){Colors.RESET}")
        print(f"\n{Colors.GREEN}All trained models saved in models/ directory{Colors.RESET}")
        print(f"{Colors.CYAN}Production-ready agent is now available for evaluation and deployment.{Colors.RESET}\n")

        return True

    def continue_training_from_model(self):
        """Continue Phase 1 or Phase 3 training from an existing checkpoint."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}╔══════════════════════════════════════════════════════════════╗{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}║           CONTINUE TRAINING FROM EXISTING MODEL              ║{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}╚══════════════════════════════════════════════════════════════╝{Colors.RESET}")

        models = detect_models_in_folder(str(self.project_dir / "models"))
        if not models:
            print(f"{Colors.RED}No models found in models/. Train a phase first.{Colors.RESET}")
            return False

        supported_models = [m for m in models if m['type'] in ('phase1', 'phase3')]
        if not supported_models:
            print(f"{Colors.RED}No Phase 1 or Phase 3 models available for continuation.{Colors.RESET}")
            print(f"{Colors.YELLOW}Train a model first or copy checkpoints into models/.{Colors.RESET}")
            return False

        selection = display_model_selection(supported_models)
        if selection < 0:
            print(f"{Colors.YELLOW}Continuation cancelled.{Colors.RESET}")
            return False

        selected_model = supported_models[selection]
        model_type = selected_model.get('type', 'unknown')
        metadata = selected_model.get('metadata') or {}

        if model_type == 'phase1':
            script_name = "train_phase1.py"
            description = "Phase 1 Continuation"
            log_file = "phase1_continue.log"
        elif model_type == 'phase3':
            script_name = "train_phase3_llm.py"
            description = "Phase 3 Continuation"
            log_file = "phase3_continue.log"
        else:
            print(f"{Colors.RED}Continuation not supported for model type: {model_type}{Colors.RESET}")
            return False

        script_path = self.src_dir / script_name
        if not script_path.exists():
            print(f"{Colors.RED}Training script missing: {script_path}{Colors.RESET}")
            return False

        print(f"\n{Colors.GREEN}Selected model: {selected_model['name']} ({model_type.upper()}){Colors.RESET}")

        # Determine run mode (test vs production)
        mode_choice = self.get_user_input(
            f"{Colors.YELLOW}Run continuation in test mode (1) or production (2)? {Colors.RESET}",
            ["1", "2"]
        )
        if mode_choice is None:
            print(f"{Colors.YELLOW}Continuation cancelled.{Colors.RESET}")
            return False
        run_test = mode_choice == "1"

        # Determine market symbol
        market = metadata.get('market')
        if market:
            print(f"{Colors.CYAN}Detected market from metadata: {market}{Colors.RESET}")
        else:
            print(f"{Colors.YELLOW}Market metadata missing. Select a dataset to continue.{Colors.RESET}")
            market = self.detect_and_select_market()
            if not market:
                print(f"{Colors.YELLOW}Continuation cancelled - market required.{Colors.RESET}")
                return False
        market = market.upper()

        command = [
            sys.executable,
            str(script_path),
            "--continue",
            "--model-path",
            selected_model['path'],
            "--market",
            market,
            "--non-interactive"
        ]

        if run_test:
            command.append("--test")

        success, _ = self.run_command_with_progress(
            command,
            f"{description} ({'Test' if run_test else 'Production'} Mode)",
            log_file
        )

        if success:
            print(f"\n{Colors.GREEN}✓ Continuation completed successfully!{Colors.RESET}")
        else:
            print(f"\n{Colors.RED}✗ Continuation failed. Review logs/{log_file}.{Colors.RESET}")

        return success

    def run_evaluation(self):
        """Evaluate the latest Phase 3 model (261D pure RL) on unseen data."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}╔══════════════════════════════════════════════════════════════╗{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}║                         EVALUATOR                              ║{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}╚══════════════════════════════════════════════════════════════╝{Colors.RESET}")

        eval_script = self.src_dir / "evaluate_phase3_llm.py"
        if not eval_script.exists():
            print(f"{Colors.RED}Error: evaluate_phase3_llm.py not found{Colors.RESET}")
            return False

        models = detect_models_in_folder(str(self.project_dir / "models"))
        phase3_models = [m for m in models if m.get('type') == 'phase3']

        if not phase3_models:
            print(f"{Colors.RED}No Phase 3 models found in models/. Run the training pipeline first.{Colors.RESET}")
            return False

        latest_model = phase3_models[0]
        metadata = latest_model.get('metadata') or {}
        market = metadata.get('market')

        if not market:
            market = self.detect_and_select_market()
            if not market:
                print(f"{Colors.YELLOW}Evaluation cancelled - no market selected.{Colors.RESET}")
                return False

        episodes = self.get_user_input(
            f"{Colors.YELLOW}Evaluation episodes (default 20): {Colors.RESET}"
        )
        episodes = episodes if episodes and episodes.isdigit() else "20"

        holdout_fraction_input = self.get_user_input(
            f"{Colors.YELLOW}Holdout fraction (0-1, default 0.2): {Colors.RESET}"
        )
        try:
            holdout_fraction = float(holdout_fraction_input) if holdout_fraction_input else 0.2
        except ValueError:
            holdout_fraction = 0.2

        confirm = self.get_user_input(
            f"{Colors.YELLOW}Evaluate {latest_model['name']} on {market} holdout data? (y/n): {Colors.RESET}",
            ["y", "n", "Y", "N"]
        )

        if confirm is None or confirm.lower() != 'y':
            print(f"{Colors.YELLOW}Evaluation cancelled.{Colors.RESET}")
            return False

        print(f"{Colors.YELLOW}Note: Phase 3 evaluation requires Phi-3-mini-4k-instruct model in project folder{Colors.RESET}\n")

        command = [
            sys.executable,
            str(eval_script),
            "--model", latest_model['path'],
            "--market", market,
            "--config", str(self.project_dir / "config" / "llm_config.yaml"),
            "--episodes", episodes,
            "--holdout-fraction", str(holdout_fraction),
            "--baseline-model", "auto"
        ]

        success, _ = self.run_command_with_progress(
            command,
            "Phase 3 Extended RL Evaluation",
            "evaluation_phase3.log"
        )

        if success:
            self.print_evaluation_results()
        else:
            print(f"\n{Colors.RED}✗ Evaluation failed. Check logs/evaluation_phase3.log{Colors.RESET}")
        return success

    def print_evaluation_results(self):
        """Display files created in results directory."""
        print(f"\n{Colors.GREEN}✓ Evaluation completed successfully!{Colors.RESET}")
        results_dir = self.project_dir / "results"
        if results_dir.exists():
            artifacts = [file.name for file in results_dir.glob("*") if file.is_file()]
            if artifacts:
                print(f"{Colors.CYAN}Results saved in results/{Colors.RESET}")
                for name in artifacts:
                    print(f"  - {name}")
    
    def show_instructions(self):
        """Show first-time user instructions."""
        instructions = f"""
{Colors.BOLD}{Colors.CYAN}╔══════════════════════════════════════════════════════════════╗{Colors.RESET}
{Colors.BOLD}{Colors.CYAN}║                    GETTING STARTED                              ║{Colors.RESET}
{Colors.BOLD}{Colors.CYAN}╚══════════════════════════════════════════════════════════════╝{Colors.RESET}

{Colors.GREEN}Welcome to RL TRAINER! This system helps you manage your{Colors.RESET}
{Colors.GREEN}reinforcement learning trading pipeline from setup to evaluation.{Colors.RESET}

{Colors.BOLD}{Colors.YELLOW}MAIN FEATURES:{Colors.RESET}
{Colors.WHITE}1. Requirements Installation{Colors.RESET}
  {Colors.CYAN}   • Installs all necessary dependencies{Colors.RESET}
  {Colors.CYAN}   • Sets up the trading environment{Colors.RESET}

{Colors.WHITE}2. Data Processing{Colors.RESET}
  {Colors.CYAN}   • Process New Data (Full): Complete data processing from .zip files{Colors.RESET}
  {Colors.CYAN}   • Update Existing Data (Incremental): Add only new dates (10x faster){Colors.RESET}
  {Colors.CYAN}   • Select from 8 trading instruments{Colors.RESET}

{Colors.WHITE}3. Training Model{Colors.RESET}
  {Colors.CYAN}   • Test Mode: Quick local testing{Colors.RESET}
  {Colors.CYAN}   • Production Mode: Full training pipeline{Colors.RESET}

{Colors.WHITE}4. Evaluator{Colors.RESET}
  {Colors.CYAN}   • Comprehensive model evaluation{Colors.RESET}
  {Colors.CYAN}   • Performance metrics and visualizations{Colors.RESET}

{Colors.BOLD}{Colors.YELLOW}SUPPORTED INSTRUMENTS:{Colors.RESET}
{Colors.WHITE}NQ  - Nasdaq 100 E-mini     ES  - S&P 500 E-mini{Colors.RESET}
{Colors.WHITE}YM  - Dow Jones E-mini      RTY - Russell 2000 E-mini{Colors.RESET}
{Colors.WHITE}MNQ - Micro Nasdaq 100      MES - Micro S&P 500{Colors.RESET}
{Colors.WHITE}M2K - Micro Russell 2000    MYM - Micro Dow Jones{Colors.RESET}

{Colors.BOLD}{Colors.YELLOW}TIPS:{Colors.RESET}
{Colors.WHITE}• All operations are logged for debugging{Colors.RESET}
{Colors.WHITE}• Press Ctrl+C to cancel any operation{Colors.RESET}
{Colors.WHITE}• Check the logs directory for detailed output{Colors.RESET}
{Colors.WHITE}• Ensure sufficient disk space for data processing{Colors.RESET}

{Colors.GREEN}Press Enter to continue to the main menu...{Colors.RESET}
"""
        print(instructions)
        input()
    
    def run(self):
        """Main menu loop."""
        # Show instructions on first run
        if not (self.logs_dir / ".instructions_shown").exists():
            self.show_instructions()
            (self.logs_dir / ".instructions_shown").touch()
        
        while True:
            try:
                self.clear_screen()
                self.display_banner()
                self.display_main_menu()
                
                choice = self.get_user_input(
                    f"{Colors.YELLOW}Select an option: {Colors.RESET}",
                    list(self.main_menu_options.keys())
                )

                if choice is None:
                    print(f"\n{Colors.YELLOW}Returning to shell...{Colors.RESET}")
                    break

                if choice == "1":
                    self.install_requirements()
                elif choice == "2":
                    self.process_data()
                elif choice == "3":
                    self.train_model()
                elif choice == "4":
                    self.run_evaluation()
                elif choice == "5":
                    print(f"\n{Colors.GREEN}Thank you for using RL TRAINER!{Colors.RESET}")
                    print(f"{Colors.CYAN}Goodbye!{Colors.RESET}")
                    break
                else:
                    print(f"{Colors.RED}Invalid option. Please try again.{Colors.RESET}")
                    time.sleep(2)
                
                if choice != "5":
                    input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.RESET}")
                    
            except KeyboardInterrupt:
                print(f"\n\n{Colors.YELLOW}Exiting RL TRAINER...{Colors.RESET}")
                break
            except Exception as e:
                print(f"\n{Colors.RED}An error occurred: {str(e)}{Colors.RESET}")
                self.logger.error(f"Menu error: {str(e)}")
                input(f"{Colors.YELLOW}Press Enter to continue...{Colors.RESET}")


def main():
    """Main entry point for the RL Trainer menu system."""
    try:
        # Create and run the menu system
        menu = RLTrainerMenu()
        menu.run()
    except Exception as e:
        print(f"{Colors.RED}Fatal error: {str(e)}{Colors.RESET}")
        logging.error(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
