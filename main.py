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
import logging
import time
import yaml
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import shutil

# Import CLI utilities (refactored from inline definitions)
from src.cli_utils import (
    Colors, clear_screen, print_header, get_user_input, prompt_confirm,
    prompt_choice, run_command_with_progress, detect_and_select_market,
    select_hardware_profile
)


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
            "3": "Hardware Stress Test & Auto-tune (JAX/PyTorch)",
            "4": "Hybrid LLM/GPU Test Run",
            "5": "Training Model (PyTorch)",
            "6": "JAX Training (Experimental)",
            "7": "Evaluator",
            "8": "Exit"
        }

        self.training_menu_options = {
            "1": "Complete Training Pipeline (Test Mode)",
            "2": "Complete Training Pipeline (Production Mode)",
            "3": "Continue Training from Existing Model",
            "4": "Back to Main Menu"
        }

        self.jax_training_menu_options = {
            "1": "Quick Validation Test (JAX Installation Check)",
            "2": "JAX Phase 1 Training (Entry Learning)",
            "3": "JAX Phase 2 Training (Position Management)",
            "4": "Custom JAX Training (Advanced)",
            "5": "Back to Main Menu"
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

    def display_main_menu(self):
        """Display the main menu options."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}╔══════════════════════════════════════════════════════════════╗{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}║                        MAIN MENU                              ║{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}╚══════════════════════════════════════════════════════════════╝{Colors.RESET}")
        print()
        
        for key, value in self.main_menu_options.items():
            if key == "8":  # Exit option
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
            instrument_input = get_user_input(
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

    def check_system_dependencies(self) -> dict:
        """
        Check for system-level dependencies (Node.js, NPM).
        
        Returns:
            Dict with 'installed' and 'missing' lists
        """
        dependencies = ['node', 'npm']
        installed = []
        missing = []
        
        for dep in dependencies:
            if shutil.which(dep):
                installed.append(dep)
            else:
                missing.append(dep)
                
        return {'installed': installed, 'missing': missing}

    def check_installed_requirements(self, check_jax: bool = False) -> dict:
        """
        Check which requirements are already installed.

        Args:
            check_jax: If True, also check JAX requirements

        Returns:
            Dict with format:
            {
                'pytorch': {'installed': [...], 'missing': [...]},
                'jax': {'installed': [...], 'missing': [...]}  # Only if check_jax=True
            }
        """
        result = {}

        # Check PyTorch requirements
        requirements_file = self.project_dir / "requirements.txt"
        installed = []
        missing = []

        if requirements_file.exists():
            with open(requirements_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if not line or line.startswith('#'):
                        continue

                    # Extract package name (handle extras like jax[cuda12])
                    package = line.split('>=')[0].split('==')[0].split('<')[0].split('[')[0].strip()

                    if self.check_package_installed(package):
                        installed.append(package)
                    else:
                        missing.append(package)

        result['pytorch'] = {'installed': installed, 'missing': missing}

        # Check JAX requirements if requested
        if check_jax:
            jax_file = self.project_dir / "requirements-jax.txt"
            jax_installed = []
            jax_missing = []

            if jax_file.exists():
                with open(jax_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue

                        # Extract package name (handle extras like jax[cuda12])
                        package = line.split('>=')[0].split('==')[0].split('<')[0].split('[')[0].strip()

                        if self.check_package_installed(package):
                            jax_installed.append(package)
                        else:
                            jax_missing.append(package)

            result['jax'] = {'installed': jax_installed, 'missing': jax_missing}

        # Check system dependencies
        result['system'] = self.check_system_dependencies()

        return result

    def _install_requirements_with_numpy_fix(self, requirements_file: Path,
                                            force_reinstall: bool = False,
                                            upgrade: bool = False) -> Tuple[bool, str]:
        """
        Install requirements with NumPy pinned first to prevent binary incompatibility.

        Args:
            requirements_file: Path to requirements.txt file
            force_reinstall: If True, force reinstall all packages
            upgrade: If True, upgrade packages to latest versions

        Returns:
            Tuple of (success, output)
        """
        print(f"\n{Colors.CYAN}Using NumPy-first installation strategy to prevent binary incompatibility...{Colors.RESET}")

        # Step 1: Install NumPy first with pinned version
        numpy_cmd = [sys.executable, "-m", "pip", "install", "numpy>=1.26.4,<2.0"]

        if force_reinstall:
            numpy_cmd.insert(4, "--force-reinstall")
        elif upgrade:
            numpy_cmd.insert(4, "--upgrade")

        print(f"\n{Colors.BOLD}Step 1/2: Installing NumPy with version constraints{Colors.RESET}")
        success, numpy_output = run_command_with_progress(
            numpy_cmd,
            "Installing NumPy (prevents binary incompatibility)",
            "numpy_install.log",
            interactive=True
        )

        if not success:
            print(f"{Colors.RED}Failed to install NumPy. Aborting installation.{Colors.RESET}")
            return False, numpy_output

        # Step 2: Install remaining requirements
        print(f"\n{Colors.BOLD}Step 2/2: Installing remaining requirements{Colors.RESET}")
        requirements_cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]

        if force_reinstall:
            requirements_cmd.insert(4, "--force-reinstall")
        elif upgrade:
            requirements_cmd.insert(4, "--upgrade")

        success, requirements_output = run_command_with_progress(
            requirements_cmd,
            f"Installing requirements from {requirements_file.name}",
            "requirements_install.log",
            interactive=True
        )

        combined_output = f"{numpy_output}\n\n{requirements_output}"
        return success, combined_output

    def install_requirements(self):
        """Install requirements using pip with smart checking and JAX support."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}╔══════════════════════════════════════════════════════════════╗{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}║                  REQUIREMENTS INSTALLATION                    ║{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}╚══════════════════════════════════════════════════════════════╝{Colors.RESET}")

        # Check if requirements files exist
        requirements_file = self.project_dir / "requirements.txt"
        jax_requirements_file = self.project_dir / "requirements-jax.txt"

        if not requirements_file.exists():
            print(f"{Colors.RED}Error: requirements.txt not found in project directory{Colors.RESET}")
            return False

        has_jax_file = jax_requirements_file.exists()

        print(f"{Colors.GREEN}Found requirements.txt at: {requirements_file}{Colors.RESET}")
        if has_jax_file:
            print(f"{Colors.GREEN}Found requirements-jax.txt at: {jax_requirements_file}{Colors.RESET}")

        # Check what's already installed (both PyTorch and JAX)
        print(f"\n{Colors.CYAN}Checking installed packages...{Colors.RESET}")
        status = self.check_installed_requirements(check_jax=has_jax_file)

        pytorch_installed = status['pytorch']['installed']
        pytorch_missing = status['pytorch']['missing']
        pytorch_total = len(pytorch_installed) + len(pytorch_missing)

        # Display PyTorch status
        print(f"\n{Colors.BOLD}PyTorch Requirements:{Colors.RESET}")
        print(f"{Colors.GREEN}  ✓ Installed: {len(pytorch_installed)}/{pytorch_total} packages{Colors.RESET}")
        if pytorch_missing:
            print(f"{Colors.YELLOW}  ✗ Missing: {len(pytorch_missing)}/{pytorch_total} packages{Colors.RESET}")

        # Display JAX status if available
        jax_installed = []
        jax_missing = []
        if has_jax_file:
            jax_installed = status['jax']['installed']
            jax_missing = status['jax']['missing']
            jax_total = len(jax_installed) + len(jax_missing)

            print(f"\n{Colors.BOLD}JAX Requirements (Experimental):{Colors.RESET}")
            print(f"{Colors.GREEN}  ✓ Installed: {len(jax_installed)}/{jax_total} packages{Colors.RESET}")
            if jax_missing:
                print(f"{Colors.YELLOW}  ✗ Missing: {len(jax_missing)}/{jax_total} packages{Colors.RESET}")

        # Display System status
        system_status = status.get('system', {'installed': [], 'missing': []})
        sys_installed = system_status['installed']
        sys_missing = system_status['missing']
        
        if sys_installed or sys_missing:
            print(f"\n{Colors.BOLD}System Dependencies (Dashboard):{Colors.RESET}")
            if sys_installed:
                print(f"{Colors.GREEN}  ✓ Installed: {', '.join(sys_installed)}{Colors.RESET}")
            if sys_missing:
                print(f"{Colors.YELLOW}  ✗ Missing: {', '.join(sys_missing)}{Colors.RESET}")

        # Determine installation options
        all_installed = not pytorch_missing and (not has_jax_file or not jax_missing) and not sys_missing

        if all_installed:
            # Everything is installed
            print(f"\n{Colors.GREEN}{Colors.BOLD}✓ All requirements are already installed!{Colors.RESET}")
            print(f"\n{Colors.CYAN}PyTorch packages:{Colors.RESET}")
            for pkg in pytorch_installed[:8]:
                print(f"  • {pkg}")
            if len(pytorch_installed) > 8:
                print(f"  ... and {len(pytorch_installed) - 8} more")

            if has_jax_file and jax_installed:
                print(f"\n{Colors.CYAN}JAX packages:{Colors.RESET}")
                for pkg in jax_installed[:5]:
                    print(f"  • {pkg}")
                if len(jax_installed) > 5:
                    print(f"  ... and {len(jax_installed) - 5} more")

            print(f"\n{Colors.YELLOW}Options:{Colors.RESET}")
            print(f"  {Colors.GREEN}1. Return to Main Menu{Colors.RESET}")
            print(f"  {Colors.CYAN}2. Reinstall PyTorch Packages{Colors.RESET}")
            if has_jax_file:
                print(f"  {Colors.CYAN}3. Reinstall JAX Packages{Colors.RESET}")
                print(f"  {Colors.CYAN}4. Reinstall Both PyTorch + JAX{Colors.RESET}")
                print(f"  {Colors.CYAN}5. Reinstall System Dependencies (Node/NPM){Colors.RESET}")
                valid_options = ["1", "2", "3", "4", "5"]
            else:
                print(f"  {Colors.CYAN}3. Reinstall System Dependencies (Node/NPM){Colors.RESET}")
                valid_options = ["1", "2", "3"]

            choice = get_user_input(
                f"\n{Colors.YELLOW}Select option: {Colors.RESET}",
                valid_options
            )

            if choice is None or choice == "1":
                print(f"{Colors.GREEN}Returning to main menu...{Colors.RESET}")
                return True
            elif choice == "2":
                # Reinstall PyTorch only
                success, _ = self._install_requirements_with_numpy_fix(requirements_file, force_reinstall=True)
            elif has_jax_file and choice == "3":
                # Reinstall JAX only
                success, _ = self._install_requirements_with_numpy_fix(jax_requirements_file, force_reinstall=True)
            elif has_jax_file and choice == "4":
                # Reinstall both
                success1, _ = self._install_requirements_with_numpy_fix(requirements_file, force_reinstall=True)
                if success1:
                    success, _ = self._install_requirements_with_numpy_fix(jax_requirements_file, force_reinstall=True)
                else:
                    success = False
            elif (has_jax_file and choice == "5") or (not has_jax_file and choice == "3"):
                # Install system deps
                cmd = ["apt-get", "update", "&&", "apt-get", "install", "-y", "nodejs", "npm"]
                # Need to run as shell command for && to work, or split it. 
                # run_command_with_progress runs directly, so we should probably run bash -c
                bash_cmd = ["bash", "-c", "apt-get update && apt-get install -y nodejs npm"]
                success, _ = run_command_with_progress(bash_cmd, "Installing Node.js and NPM", "system_install.log")
            else:
                return True

        else:
            # Some packages are missing
            if pytorch_missing:
                print(f"\n{Colors.YELLOW}Missing PyTorch packages:{Colors.RESET}")
                for pkg in pytorch_missing[:10]:
                    print(f"  • {pkg}")
                if len(pytorch_missing) > 10:
                    print(f"  ... and {len(pytorch_missing) - 10} more")

            if has_jax_file and jax_missing:
                print(f"\n{Colors.YELLOW}Missing JAX packages:{Colors.RESET}")
                for pkg in jax_missing[:10]:
                    print(f"  • {pkg}")
                if len(jax_missing) > 10:
                    print(f"  ... and {len(jax_missing) - 10} more")

            print(f"\n{Colors.YELLOW}Installation Options:{Colors.RESET}")
            print(f"  {Colors.GREEN}1. Install PyTorch Requirements Only{Colors.RESET}")
            if has_jax_file:
                print(f"  {Colors.GREEN}2. Install JAX Requirements Only{Colors.RESET}")
                print(f"  {Colors.CYAN}3. Install Both PyTorch + JAX{Colors.RESET}")
                print(f"  {Colors.CYAN}4. Install System Dependencies (Node/NPM){Colors.RESET}")
                print(f"  {Colors.RED}5. Cancel / Return to Main Menu{Colors.RESET}")
                valid_options = ["1", "2", "3", "4", "5"]
                cancel_option = "5"
            else:
                print(f"  {Colors.CYAN}2. Install System Dependencies (Node/NPM){Colors.RESET}")
                print(f"  {Colors.RED}3. Cancel / Return to Main Menu{Colors.RESET}")
                valid_options = ["1", "2", "3"]
                cancel_option = "3"

            choice = get_user_input(
                f"\n{Colors.YELLOW}Select option: {Colors.RESET}",
                valid_options
            )

            if choice is None or choice == cancel_option:
                print(f"{Colors.YELLOW}Installation cancelled. Returning to main menu...{Colors.RESET}")
                return False
            elif choice == "1":
                # Install PyTorch only
                success, _ = self._install_requirements_with_numpy_fix(requirements_file)
            elif has_jax_file and choice == "2":
                # Install JAX only
                success, _ = self._install_requirements_with_numpy_fix(jax_requirements_file)
            elif has_jax_file and choice == "3":
                # Install both
                success1, _ = self._install_requirements_with_numpy_fix(requirements_file)
                if success1:
                    success, _ = self._install_requirements_with_numpy_fix(jax_requirements_file)
                else:
                    success = False
            elif (has_jax_file and choice == "4") or (not has_jax_file and choice == "2"):
                # Install system deps
                print(f"\n{Colors.YELLOW}Note: This requires root/sudo privileges (common in Runpod/Docker).{Colors.RESET}")
                bash_cmd = ["bash", "-c", "apt-get update && apt-get install -y nodejs npm"]
                success, _ = run_command_with_progress(bash_cmd, "Installing Node.js and NPM", "system_install.log")
            else:
                return False

        if success:
            print(f"\n{Colors.GREEN}✓ Installation completed successfully!{Colors.RESET}")
            print(f"{Colors.CYAN}All requested dependencies are now installed.{Colors.RESET}")

            # Restart program to ensure new NumPy/package versions are loaded
            print(f"\n{Colors.YELLOW}Requirements updated successfully. Restarting program to apply changes...{Colors.RESET}")
            time.sleep(3)

            # Restart the program with the same arguments
            os.execv(sys.executable, [sys.executable] + sys.argv)
        else:
            print(f"\n{Colors.RED}✗ Installation failed. Check logs for details.{Colors.RESET}")
            print(f"{Colors.YELLOW}Tip: Try running installation commands manually{Colors.RESET}")

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

        choice = get_user_input(
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
        confirm = get_user_input(
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
        success, output = run_command_with_progress(
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

        # Run with interactive mode to allow user prompts
        command = [sys.executable, str(incremental_script), "--market", instrument]

        success, _ = run_command_with_progress(
            command,
            f"Incremental Data Update ({instrument})",
            "incremental_update.log",
            interactive=True
        )

        if success:
            print(f"\n{Colors.GREEN}✓ Incremental update completed for {instrument}!{Colors.RESET}")
            print(f"{Colors.CYAN}Existing data has been updated with new dates.{Colors.RESET}")
        else:
            print(f"\n{Colors.YELLOW}Incremental update exited with errors. Check logs/incremental_update.log{Colors.RESET}")

        return success
    
    def train_model(self):
        """Handle model training with test/production options."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}╔══════════════════════════════════════════════════════════════╗{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}║                      TRAINING MODEL                             ║{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}╚══════════════════════════════════════════════════════════════╝{Colors.RESET}")

        while True:
            self.display_training_menu()
            choice = get_user_input(
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
        confirm = get_user_input(
            f"{Colors.YELLOW}Proceed with complete pipeline test? (y/n): {Colors.RESET}",
            ["y", "n", "Y", "N"]
        )

        if confirm is None or confirm.lower() != 'y':
            print(f"{Colors.YELLOW}Pipeline test cancelled.{Colors.RESET}")
            return False

        # Market selection - ONCE for entire pipeline
        selected_market = detect_and_select_market(self.project_dir)
        if selected_market is None:
            print(f"{Colors.YELLOW}Pipeline cancelled - no market selected.{Colors.RESET}")
            return False

        # Hardware Profile Selection
        hardware_profile = select_hardware_profile(self.project_dir)
        if hardware_profile:
            print(f"{Colors.GREEN}Using hardware profile: {Path(hardware_profile).name}{Colors.RESET}")

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
        if hardware_profile:
            command.extend(["--hardware-profile", hardware_profile])

        success, output = run_command_with_progress(
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
        if hardware_profile:
            command.extend(["--hardware-profile", hardware_profile])

        success, output = run_command_with_progress(
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
                    proceed = get_user_input(
                        f"{Colors.YELLOW}Continue with Phase 3? (y/n): {Colors.RESET}",
                        ["y", "n", "Y", "N"]
                    )
                    if proceed is None or proceed.lower() != 'y':
                        print(f"{Colors.YELLOW}Phase 3 skipped. Pipeline completed through Phase 2.{Colors.RESET}")
                        return True
            else:
                print(f"{Colors.YELLOW}⚠ No GPU detected. LLM inference will be slow.{Colors.RESET}")
                proceed = get_user_input(
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
        if hardware_profile:
            command.extend(["--hardware-profile", hardware_profile])

        success, output = run_command_with_progress(
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
        confirm = get_user_input(
            f"{Colors.YELLOW}Proceed with complete production pipeline? (y/n): {Colors.RESET}",
            ["y", "n", "Y", "N"]
        )

        if confirm is None or confirm.lower() != 'y':
            print(f"{Colors.YELLOW}Production pipeline cancelled.{Colors.RESET}")
            return False

        # Market selection - ONCE for entire pipeline
        selected_market = detect_and_select_market(self.project_dir)
        if selected_market is None:
            print(f"{Colors.YELLOW}Pipeline cancelled - no market selected.{Colors.RESET}")
            return False

        # Hardware Profile Selection
        hardware_profile = select_hardware_profile(self.project_dir)
        if hardware_profile:
            print(f"{Colors.GREEN}Using hardware profile: {Path(hardware_profile).name}{Colors.RESET}")

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
        if hardware_profile:
            command.extend(["--hardware-profile", hardware_profile])

        success, output = run_command_with_progress(
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
        if hardware_profile:
            command.extend(["--hardware-profile", hardware_profile])

        success, output = run_command_with_progress(
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
                    proceed = get_user_input(
                        f"{Colors.YELLOW}Continue with Phase 3? (y/n): {Colors.RESET}",
                        ["y", "n", "Y", "N"]
                    )
                    if proceed is None or proceed.lower() != 'y':
                        print(f"{Colors.YELLOW}Phase 3 skipped. Pipeline completed through Phase 2.{Colors.RESET}")
                        return True
            else:
                print(f"{Colors.RED}✗ No GPU detected. Phase 3 requires GPU for production training.{Colors.RESET}")
                proceed = get_user_input(
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
        if hardware_profile:
            command.extend(["--hardware-profile", hardware_profile])

        success, output = run_command_with_progress(
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

        # Lazy import to avoid bootstrap dependency issues
        from src.model_utils import detect_models_in_folder, display_model_selection

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
        mode_choice = get_user_input(
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
            market = detect_and_select_market(self.project_dir)
            if not market:
                print(f"{Colors.YELLOW}Continuation cancelled - market required.{Colors.RESET}")
                return False
        market = market.upper()

        # Hardware Profile Selection
        hardware_profile = select_hardware_profile(self.project_dir)
        if hardware_profile:
            print(f"{Colors.GREEN}Using hardware profile: {Path(hardware_profile).name}{Colors.RESET}")

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

        if hardware_profile:
            command.extend(["--hardware-profile", hardware_profile])

        success, _ = run_command_with_progress(
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

        # Lazy import to avoid bootstrap dependency issues
        from src.model_utils import detect_models_in_folder

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
            market = detect_and_select_market(self.project_dir)
            if not market:
                print(f"{Colors.YELLOW}Evaluation cancelled - no market selected.{Colors.RESET}")
                return False

        episodes = get_user_input(
            f"{Colors.YELLOW}Evaluation episodes (default 20): {Colors.RESET}"
        )
        episodes = episodes if episodes and episodes.isdigit() else "20"

        holdout_fraction_input = get_user_input(
            f"{Colors.YELLOW}Holdout fraction (0-1, default 0.2): {Colors.RESET}"
        )
        try:
            holdout_fraction = float(holdout_fraction_input) if holdout_fraction_input else 0.2
        except ValueError:
            holdout_fraction = 0.2

        confirm = get_user_input(
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

        success, _ = run_command_with_progress(
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

    def run_stress_test(self):
        """Run hardware stress test and auto-tuning with interactive menu."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}╔══════════════════════════════════════════════════════════════╗{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}║              HARDWARE STRESS TEST & AUTO-TUNE                  ║{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}╚══════════════════════════════════════════════════════════════╝{Colors.RESET}")

        print(f"\n{Colors.YELLOW}Test your hardware and find optimal training configurations.{Colors.RESET}\n")

        # Check which stress test scripts are available
        jax_stress = self.project_dir / "scripts" / "stress_hardware_jax.py"
        autotune_stress = self.project_dir / "scripts" / "stress_hardware_autotune.py"

        if not jax_stress.exists() and not autotune_stress.exists():
            print(f"{Colors.RED}No stress test scripts found in scripts/ directory.{Colors.RESET}")
            return False

        # Build menu options dynamically based on available scripts
        stress_menu_options = {}
        option_num = 1

        if jax_stress.exists():
            stress_menu_options[str(option_num)] = "JAX Phase 1 Stress Test (Entry Signal Learning) - Recommended"
            option_num += 1
            stress_menu_options[str(option_num)] = "JAX Phase 2 Stress Test (Position Management) - Recommended"
            option_num += 1

        if autotune_stress.exists():
            stress_menu_options[str(option_num)] = "Legacy PyTorch Phase 3 Autotune (LLM/Hybrid) - Advanced Only"
            option_num += 1

        if jax_stress.exists():
            stress_menu_options[str(option_num)] = "Validate Existing Profile"
            option_num += 1

        stress_menu_options[str(option_num)] = "Back to Main Menu"

        # Display stress test type selection menu
        print(f"{Colors.BOLD}SELECT STRESS TEST TYPE:{Colors.RESET}")
        for key, desc in stress_menu_options.items():
            # Highlight recommended options
            if "Recommended" in desc:
                print(f"{Colors.GREEN}  {key}. {desc}{Colors.RESET}")
            elif "Advanced Only" in desc:
                print(f"{Colors.YELLOW}  {key}. {desc}{Colors.RESET}")
            else:
                print(f"{Colors.CYAN}  {key}. {desc}{Colors.RESET}")

        choice = get_user_input(
            f"\nSelect option (1-{len(stress_menu_options)}): ",
            list(stress_menu_options.keys())
        )

        if choice is None or stress_menu_options[choice] == "Back to Main Menu":
            print(f"{Colors.YELLOW}Returning to main menu...{Colors.RESET}")
            return False

        selected_option = stress_menu_options[choice]

        # Handle JAX Phase 1 or Phase 2 stress test
        if "JAX Phase" in selected_option:
            phase = 1 if "Phase 1" in selected_option else 2
            return self._run_jax_stress_test(jax_stress, phase)

        # Handle Profile Validation
        elif "Validate Existing Profile" in selected_option:
            return self._validate_hardware_profile(jax_stress)

        # Handle Legacy PyTorch Phase 3 Autotune
        elif "Legacy PyTorch" in selected_option:
            print(f"\n{Colors.YELLOW}Warning: This uses Phase 3 LLM (PyTorch), which requires significant VRAM.{Colors.RESET}")
            print(f"{Colors.YELLOW}For most users, JAX Phase 1/2 tests are recommended.{Colors.RESET}\n")

            if not prompt_confirm("Continue with Phase 3 autotune?", default_yes=False):
                return False

            print(f"\n{Colors.CYAN}Running: stress_hardware_autotune.py{Colors.RESET}\n")

            success, _ = run_command_with_progress(
                [sys.executable, str(autotune_stress)],
                "Phase 3 Hardware Autotune",
                "stress_test_phase3.log"
            )

            if success:
                print(f"\n{Colors.GREEN}✓ Phase 3 autotune completed!{Colors.RESET}")
                print(f"{Colors.CYAN}Check logs/stress_test_phase3.log for results.{Colors.RESET}")
            else:
                print(f"\n{Colors.RED}✗ Autotune failed. Check logs/stress_test_phase3.log{Colors.RESET}")

            return success

        return False

    def _run_jax_stress_test(self, jax_stress: Path, phase: int) -> bool:
        """
        Run JAX stress test for specified phase with user configuration.

        Args:
            jax_stress: Path to JAX stress test script
            phase: Training phase (1 or 2)

        Returns:
            True if successful, False otherwise
        """
        # Select test intensity
        intensity_options = {
            "1": "Quick (5 runs, ~10-15 min) - Fast hardware check",
            "2": "Standard (10 runs, ~20-30 min) - Balanced",
            "3": "Thorough (20 runs, ~40-60 min) - Complete optimization"
        }

        print(f"\n{Colors.BOLD}SELECT TEST INTENSITY:{Colors.RESET}")
        for key, desc in intensity_options.items():
            print(f"{Colors.CYAN}  {key}. {desc}{Colors.RESET}")

        intensity_choice = get_user_input(
            "\nSelect intensity (1-3): ",
            list(intensity_options.keys())
        )

        if intensity_choice is None:
            print(f"{Colors.YELLOW}Cancelled.{Colors.RESET}")
            return False

        # Map intensity to max_runs
        max_runs_map = {"1": 5, "2": 10, "3": 20}
        max_runs = max_runs_map[intensity_choice]

        # Select data type
        data_options = {
            "1": "Dummy Data (fast, hardware testing only)",
            "2": "Real Market Data (slower, includes quality validation) - Recommended"
        }

        print(f"\n{Colors.BOLD}SELECT DATA TYPE:{Colors.RESET}")
        for key, desc in data_options.items():
            if "Recommended" in desc:
                print(f"{Colors.GREEN}  {key}. {desc}{Colors.RESET}")
            else:
                print(f"{Colors.CYAN}  {key}. {desc}{Colors.RESET}")

        data_choice = get_user_input(
            "\nSelect data type (1-2): ",
            list(data_options.keys())
        )

        if data_choice is None:
            print(f"{Colors.YELLOW}Cancelled.{Colors.RESET}")
            return False

        use_real_data = (data_choice == "2")

        # Select market
        market = "TEST"  # Default for dummy data
        if use_real_data:
            market = detect_and_select_market(self.project_dir)
            if not market:
                print(f"{Colors.YELLOW}No market selected. Using dummy data instead.{Colors.RESET}")
                use_real_data = False
                market = "TEST"

        # Display configuration summary
        phase_name = "Entry Signal Learning" if phase == 1 else "Position Management"
        time_estimate = {
            "1": "10-15 minutes",
            "2": "20-30 minutes",
            "3": "40-60 minutes"
        }[intensity_choice]

        print(f"\n{Colors.BOLD}TEST CONFIGURATION:{Colors.RESET}")
        print(f"{Colors.CYAN}  Phase: {phase} ({phase_name}){Colors.RESET}")
        print(f"{Colors.CYAN}  Market: {market}{Colors.RESET}")
        print(f"{Colors.CYAN}  Max Runs: {max_runs}{Colors.RESET}")
        print(f"{Colors.CYAN}  Data: {'Real market data' if use_real_data else 'Dummy data'}{Colors.RESET}")
        print(f"{Colors.CYAN}  Estimated Time: {time_estimate}{Colors.RESET}\n")

        # Confirm before running
        if not prompt_confirm(f"Start JAX Phase {phase} stress test?", default_yes=True):
            print(f"{Colors.YELLOW}Cancelled.{Colors.RESET}")
            return False

        # Build command
        command = [
            sys.executable,
            str(jax_stress),
            "--phase", str(phase),
            "--market", market,
            "--max-runs", str(max_runs),
            "--patience", "3",
            "--min-gain", "0.5"
        ]

        if use_real_data:
            command.append("--use-real-data")

        # Run the stress test
        print(f"\n{Colors.CYAN}Running JAX Phase {phase} Stress Test...{Colors.RESET}\n")

        success, _ = run_command_with_progress(
            command,
            f"JAX Phase {phase} Stress Test",
            f"stress_test_jax_phase{phase}.log"
        )

        if success:
            self._display_stress_test_results(phase)
        else:
            print(f"\n{Colors.RED}✗ Stress test failed.{Colors.RESET}")
            print(f"{Colors.CYAN}Check logs/stress_test_jax_phase{phase}.log for details.{Colors.RESET}")

        return success

    def _validate_hardware_profile(self, jax_stress: Path) -> bool:
        """
        Validate an existing hardware profile.

        Args:
            jax_stress: Path to JAX stress test script

        Returns:
            True if successful, False otherwise
        """
        profiles_dir = self.project_dir / "config" / "hardware_profiles"

        if not profiles_dir.exists():
            print(f"\n{Colors.RED}Hardware profiles directory not found: {profiles_dir}{Colors.RESET}")
            print(f"{Colors.YELLOW}Run a stress test first to generate profiles.{Colors.RESET}")
            return False

        # List available profiles
        profile_files = list(profiles_dir.glob("*.yaml"))

        if not profile_files:
            print(f"\n{Colors.RED}No hardware profiles found in {profiles_dir}{Colors.RESET}")
            print(f"{Colors.YELLOW}Run a stress test first to generate profiles.{Colors.RESET}")
            return False

        print(f"\n{Colors.BOLD}AVAILABLE HARDWARE PROFILES:{Colors.RESET}")
        profiles_map = {}
        for idx, profile_file in enumerate(profile_files, 1):
            profiles_map[str(idx)] = profile_file
            print(f"{Colors.CYAN}  {idx}. {profile_file.name}{Colors.RESET}")

        profiles_map[str(len(profile_files) + 1)] = None  # Cancel option
        print(f"{Colors.CYAN}  {len(profile_files) + 1}. Cancel{Colors.RESET}")

        choice = get_user_input(
            f"\nSelect profile to validate (1-{len(profiles_map)}): ",
            list(profiles_map.keys())
        )

        if choice is None or profiles_map[choice] is None:
            print(f"{Colors.YELLOW}Cancelled.{Colors.RESET}")
            return False

        selected_profile = profiles_map[choice]

        print(f"\n{Colors.CYAN}Validating profile: {selected_profile.name}{Colors.RESET}\n")

        command = [
            sys.executable,
            str(jax_stress),
            "--validate-profile", str(selected_profile)
        ]

        success, _ = run_command_with_progress(
            command,
            "Profile Validation",
            "profile_validation.log"
        )

        if success:
            print(f"\n{Colors.GREEN}✓ Profile validation completed!{Colors.RESET}")
            print(f"{Colors.CYAN}Check logs/profile_validation.log for details.{Colors.RESET}")
        else:
            print(f"\n{Colors.RED}✗ Profile validation failed.{Colors.RESET}")
            print(f"{Colors.CYAN}Check logs/profile_validation.log for details.{Colors.RESET}")

        return success

    def _display_stress_test_results(self, phase: int):
        """
        Display stress test results and next steps.

        Args:
            phase: Training phase that was tested
        """
        print(f"\n{Colors.GREEN}✓ Stress test completed!{Colors.RESET}\n")

        profiles_dir = self.project_dir / "config" / "hardware_profiles"
        results_dir = self.project_dir / "results"

        # Show generated profiles
        if profiles_dir.exists():
            profile_files = list(profiles_dir.glob("*.yaml"))
            if profile_files:
                print(f"{Colors.BOLD}PROFILES SAVED TO:{Colors.RESET}")
                print(f"{Colors.CYAN}  {profiles_dir}/{Colors.RESET}")
                for profile_file in profile_files:
                    print(f"{Colors.GREEN}    - {profile_file.name}{Colors.RESET}")
                print()

        # Show CSV log if available
        if results_dir.exists():
            csv_files = sorted(results_dir.glob("jax_stress_test_*.csv"))
            if csv_files:
                latest_csv = csv_files[-1]
                print(f"{Colors.BOLD}DETAILED LOG:{Colors.RESET}")
                print(f"{Colors.CYAN}  {latest_csv}{Colors.RESET}\n")

        # Show usage instructions
        print(f"{Colors.BOLD}TO USE THE BEST PROFILE IN TRAINING:{Colors.RESET}")

        if phase == 1:
            print(f"{Colors.YELLOW}  python src/jax_migration/train_ppo_jax_fixed.py \\{Colors.RESET}")
            print(f"{Colors.YELLOW}    --profile config/hardware_profiles/balanced.yaml{Colors.RESET}")
        else:
            print(f"{Colors.YELLOW}  python src/jax_migration/train_phase2_jax.py \\{Colors.RESET}")
            print(f"{Colors.YELLOW}    --profile config/hardware_profiles/balanced.yaml{Colors.RESET}")

        print()
        print(f"{Colors.CYAN}Tip: Use 'balanced.yaml' for general training, 'max_gpu.yaml' for speed,{Colors.RESET}")
        print(f"{Colors.CYAN}     or 'max_quality.yaml' for best results (slower).{Colors.RESET}\n")

        input(f"{Colors.YELLOW}Press Enter to continue...{Colors.RESET}")

    def run_hybrid_test(self):
        """Run hardware-maximized hybrid LLM/GPU validation test."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}╔══════════════════════════════════════════════════════════════╗{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}║           HYBRID LLM/GPU VALIDATION TEST                       ║{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}╚══════════════════════════════════════════════════════════════╝{Colors.RESET}")
        print(f"\n{Colors.YELLOW}Hardware-maximized validation combining LLM reasoning with GPU-accelerated RL.{Colors.RESET}\n")

        # Check if the hybrid test script exists
        test_script = self.project_dir / "scripts" / "run_hybrid_test.py"
        if not test_script.exists():
            print(f"{Colors.RED}Hybrid test script not found: {test_script}{Colors.RESET}")
            return False

        # Market selection
        market = detect_and_select_market(self.project_dir)
        if not market:
            return False

        # Preset selection
        print(f"\n{Colors.BOLD}{Colors.YELLOW}SELECT VALIDATION PRESET:{Colors.RESET}")
        preset_options = {
            "1": ("Fast - Quick validation (5% timesteps, 12 envs, ~15-20 min)", "fast"),
            "2": ("Heavy - Thorough validation (15% timesteps, 24 envs, ~45-60 min)", "heavy")
        }

        for key, (desc, _) in preset_options.items():
            print(f"{Colors.CYAN}  {key}. {desc}{Colors.RESET}")

        preset_choice = get_user_input(
            f"{Colors.YELLOW}Select preset (1-2): {Colors.RESET}",
            list(preset_options.keys())
        )

        if preset_choice is None:
            print(f"{Colors.YELLOW}Test cancelled.{Colors.RESET}")
            return False

        preset = preset_options[preset_choice][1]

        # Display configuration and confirm
        print(f"\n{Colors.BOLD}{Colors.GREEN}TEST CONFIGURATION:{Colors.RESET}")
        print(f"{Colors.CYAN}  Market: {market}{Colors.RESET}")
        print(f"{Colors.CYAN}  Preset: {preset}{Colors.RESET}")
        print()

        confirm = prompt_confirm("Start hybrid LLM/GPU validation test?", default_yes=True)
        if not confirm:
            print(f"{Colors.YELLOW}Test cancelled.{Colors.RESET}")
            return False

        print(f"\n{Colors.CYAN}Starting hybrid validation test...{Colors.RESET}\n")

        # Run the hybrid test script
        command = [
            sys.executable, str(test_script),
            "--market", market,
            "--preset", preset
        ]

        success, _ = run_command_with_progress(
            command,
            f"Hybrid LLM/GPU Validation ({market}, {preset} preset)",
            f"hybrid_test_{market.lower()}_{preset}.log"
        )

        if success:
            print(f"\n{Colors.GREEN}✓ Hybrid validation test completed!{Colors.RESET}")
        else:
            print(f"\n{Colors.RED}✗ Hybrid validation test failed.{Colors.RESET}")

        return success

    def run_jax_training_menu(self):
        """JAX training submenu."""
        while True:
            clear_screen()
            print(f"\n{Colors.BOLD}{Colors.CYAN}╔══════════════════════════════════════════════════════════════╗{Colors.RESET}")
            print(f"{Colors.BOLD}{Colors.CYAN}║              JAX TRAINING (EXPERIMENTAL)                       ║{Colors.RESET}")
            print(f"{Colors.BOLD}{Colors.CYAN}╚══════════════════════════════════════════════════════════════╝{Colors.RESET}")
            print()

            for key, value in self.jax_training_menu_options.items():
                color = Colors.YELLOW if key == "5" else Colors.GREEN
                print(f"{color}  {key}. {value}{Colors.RESET}")
            print()

            choice = get_user_input(
                f"{Colors.YELLOW}Select JAX option: {Colors.RESET}",
                list(self.jax_training_menu_options.keys())
            )

            if choice is None or choice == "5":
                break

            if choice == "1":
                self.run_jax_quickstart()
            elif choice == "2":
                self.run_jax_phase1()
            elif choice == "3":
                self.run_jax_phase2()
            elif choice == "4":
                self.run_custom_jax_training()

            # Pause before returning to menu so user can see results
            print()
            input(f"{Colors.YELLOW}Press Enter to continue...{Colors.RESET}")

    def run_jax_quickstart(self):
        """Run JAX installation validation."""
        script = self.src_dir / "jax_migration" / "quickstart.py"
        if not script.exists():
            print(f"{Colors.RED}JAX quickstart script not found: {script}{Colors.RESET}")
            return False

        success, _ = run_command_with_progress(
            [sys.executable, str(script)],
            "JAX Installation Validation",
            "jax_quickstart.log"
        )
        return success

    def run_jax_phase1(self):
        """Run JAX Phase 1 training."""
        market = detect_and_select_market(self.project_dir)
        if not market:
            return False

        # Hardware Profile Selection
        hardware_profile = select_hardware_profile(self.project_dir)
        num_envs = None

        if hardware_profile:
            print(f"{Colors.GREEN}Using hardware profile: {Path(hardware_profile).name}{Colors.RESET}")
            try:
                with open(hardware_profile, 'r') as f:
                    profile_data = yaml.safe_load(f)
                    if 'num_envs' in profile_data:
                        num_envs = int(profile_data['num_envs'])
                        print(f"{Colors.CYAN}  - Loaded num_envs: {num_envs}{Colors.RESET}")
            except Exception as e:
                print(f"{Colors.RED}Failed to load profile: {e}{Colors.RESET}")
                hardware_profile = None

        if num_envs is None:
            env_options = {
                "1": 512,
                "2": 1024,
                "3": 2048,
                "4": 4096
            }
            print(f"\n{Colors.BOLD}Choose number of environments:{Colors.RESET}")
            print(f"{Colors.CYAN}  1. 512   (conservative, ~4GB){Colors.RESET}")
            print(f"{Colors.CYAN}  2. 1024  (balanced, ~6GB){Colors.RESET}")
            print(f"{Colors.CYAN}  3. 2048  (high performance, ~10GB){Colors.RESET}")
            print(f"{Colors.CYAN}  4. 4096  (max throughput, ~18GB){Colors.RESET}")

            env_choice = get_user_input(
                f"{Colors.YELLOW}Select environment count (1-4): {Colors.RESET}",
                list(env_options.keys())
            )
            if env_choice is None:
                return False
            num_envs = env_options[env_choice]

        timestep_options = {
            "1": 500_000,
            "2": 2_000_000,
            "3": 5_000_000,
            "4": 10_000_000,
            "5": 20_000_000
        }
        print(f"\n{Colors.BOLD}Choose total timesteps:{Colors.RESET}")
        print(f"{Colors.CYAN}  1. 500,000      (quick test - ~30 sec){Colors.RESET}")
        print(f"{Colors.CYAN}  2. 2,000,000    (baseline - ~2 min, foundation quality){Colors.RESET}")
        print(f"{Colors.CYAN}  3. 5,000,000    (balanced - ~5 min, good foundation){Colors.RESET}")
        print(f"{Colors.CYAN}  4. 10,000,000   (recommended - ~10 min, strong foundation){Colors.RESET}")
        print(f"{Colors.CYAN}  5. 20,000,000   (extended - ~20 min, excellent foundation){Colors.RESET}")

        ts_choice = get_user_input(
            f"{Colors.YELLOW}Select total timesteps (1-5): {Colors.RESET}",
            list(timestep_options.keys())
        )
        if ts_choice is None:
            return False
        timesteps = timestep_options[ts_choice]

        confirm = get_user_input(
            f"{Colors.YELLOW}Start JAX Phase 1 for {market} with {num_envs} envs and {timesteps:,} steps? (y/n): {Colors.RESET}",
            ["y", "n", "Y", "N"]
        )
        if confirm is None or confirm.lower() != "y":
            print(f"{Colors.YELLOW}JAX Phase 1 cancelled.{Colors.RESET}")
            return False

        # Determine which module to use
        script_fixed = self.src_dir / "jax_migration" / "train_ppo_jax_fixed.py"
        script_normal = self.src_dir / "jax_migration" / "train_ppo_jax.py"

        if script_fixed.exists():
            module_name = "src.jax_migration.train_ppo_jax_fixed"
        elif script_normal.exists():
            module_name = "src.jax_migration.train_ppo_jax"
        else:
            print(f"{Colors.RED}JAX Phase 1 script not found{Colors.RESET}")
            return False

        print(f"\n{Colors.CYAN}Starting JAX Phase 1 training for {market}...{Colors.RESET}\n")

        # Run as module to support relative imports
        command = [
            sys.executable, "-m", module_name,
            "--market", market,
            "--num_envs", str(num_envs),
            "--total_timesteps", str(timesteps),
            "--data_path", str(self.project_dir / "data" / f"{market}_D1M.csv")
        ]

        success, _ = run_command_with_progress(
            command,
            f"JAX Phase 1 Training ({market}, {num_envs} envs, {timesteps:,} steps)",
            f"jax_phase1_{market.lower()}.log"
        )
        return success

    def run_jax_phase2(self):
        """Run JAX Phase 2 training with hardware profile support."""
        market = detect_and_select_market(self.project_dir)
        if not market:
            return False

        script = self.src_dir / "jax_migration" / "train_phase2_jax.py"
        if not script.exists():
            print(f"{Colors.RED}JAX Phase 2 script not found: {script}{Colors.RESET}")
            return False

        # Hardware Profile Selection
        hardware_profile = select_hardware_profile(self.project_dir)
        num_envs = None

        if hardware_profile:
            print(f"{Colors.GREEN}Using hardware profile: {Path(hardware_profile).name}{Colors.RESET}")
            try:
                with open(hardware_profile, 'r') as f:
                    profile_data = yaml.safe_load(f)
                    if 'num_envs' in profile_data:
                        num_envs = int(profile_data['num_envs'])
                        print(f"{Colors.CYAN}  - Loaded num_envs: {num_envs}{Colors.RESET}")
            except Exception as e:
                print(f"{Colors.RED}Failed to load profile: {e}{Colors.RESET}")
                hardware_profile = None

        if num_envs is None:
            # Prompt user to select num_envs manually
            print(f"\n{Colors.BOLD}{Colors.YELLOW}SELECT NUMBER OF PARALLEL ENVIRONMENTS:{Colors.RESET}")
            env_options = {
                "1": ("512 envs", 512),
                "2": ("1024 envs", 1024),
                "3": ("2048 envs (Recommended)", 2048),
                "4": ("4096 envs (High-end hardware)", 4096)
            }

            for key, (desc, _) in env_options.items():
                print(f"{Colors.CYAN}  {key}. {desc}{Colors.RESET}")

            env_choice = get_user_input(
                f"{Colors.YELLOW}Select option (1-4): {Colors.RESET}",
                list(env_options.keys())
            )

            if env_choice is None:
                print(f"{Colors.YELLOW}Training cancelled.{Colors.RESET}")
                return False

            num_envs = env_options[env_choice][1]

        # Prompt for timesteps
        print(f"\n{Colors.BOLD}{Colors.YELLOW}SELECT TRAINING DURATION:{Colors.RESET}")
        timestep_options = {
            "1": ("Quick Test - 500K steps (~5-10 min)", 500_000),
            "2": ("Short - 5M steps (~1 hour, 2.5x Phase 1)", 5_000_000),
            "3": ("Standard - 10M steps (~2 hours, matches Phase 1)", 10_000_000),
            "4": ("Extended - 25M steps (~5 hours, 2.5x Phase 1)", 25_000_000),
            "5": ("Production - 50M steps (~10 hours, 5x Phase 1)", 50_000_000),
            "6": ("Maximum - 100M steps (~20 hours, 10x Phase 1)", 100_000_000)
        }

        for key, (desc, _) in timestep_options.items():
            print(f"{Colors.CYAN}  {key}. {desc}{Colors.RESET}")

        timestep_choice = get_user_input(
            f"{Colors.YELLOW}Select option (1-6): {Colors.RESET}",
            list(timestep_options.keys())
        )

        if timestep_choice is None:
            print(f"{Colors.YELLOW}Training cancelled.{Colors.RESET}")
            return False

        timesteps = timestep_options[timestep_choice][1]

        # Display configuration and confirm
        print(f"\n{Colors.BOLD}{Colors.GREEN}TRAINING CONFIGURATION:{Colors.RESET}")
        print(f"{Colors.CYAN}  Market: {market}{Colors.RESET}")
        print(f"{Colors.CYAN}  Parallel Environments: {num_envs}{Colors.RESET}")
        print(f"{Colors.CYAN}  Total Timesteps: {timesteps:,}{Colors.RESET}")
        if hardware_profile:
            print(f"{Colors.CYAN}  Hardware Profile: {Path(hardware_profile).name}{Colors.RESET}")
        print()

        confirm = prompt_confirm("Start JAX Phase 2 training with these settings?", default_yes=True)
        if not confirm:
            print(f"{Colors.YELLOW}Training cancelled.{Colors.RESET}")
            return False

        print(f"\n{Colors.CYAN}Starting JAX Phase 2 training for {market}...{Colors.RESET}\n")

        # Run as module to support relative imports
        command = [
            sys.executable, "-m", "src.jax_migration.train_phase2_jax",
            "--num_envs", str(num_envs),
            "--total_timesteps", str(timesteps),
            "--data_path", str(self.project_dir / "data" / f"{market}_D1M.csv"),
            "--checkpoint_dir", str(self.project_dir / "models" / f"phase2_jax_{market.lower()}")
        ]

        success, _ = run_command_with_progress(
            command,
            f"JAX Phase 2 Training ({market})",
            f"jax_phase2_{market.lower()}.log"
        )
        return success

    def run_custom_jax_training(self):
        """Run custom JAX training with hardware profile support."""
        print(f"\n{Colors.BOLD}{Colors.YELLOW}CUSTOM JAX TRAINING{Colors.RESET}\n")

        market = detect_and_select_market(self.project_dir)
        if not market:
            return False

        # Hardware Profile Selection
        hardware_profile = select_hardware_profile(self.project_dir)
        default_envs = 1024
        default_steps = 2_000_000

        if hardware_profile:
            print(f"{Colors.GREEN}Using hardware profile: {Path(hardware_profile).name}{Colors.RESET}")
            try:
                with open(hardware_profile, 'r') as f:
                    profile_data = yaml.safe_load(f)
                    if 'num_envs' in profile_data:
                        default_envs = int(profile_data['num_envs'])
                        print(f"{Colors.CYAN}  - Loaded num_envs default: {default_envs}{Colors.RESET}")
                    if 'total_timesteps' in profile_data:
                        default_steps = int(profile_data['total_timesteps'])
                        print(f"{Colors.CYAN}  - Loaded timesteps default: {default_steps}{Colors.RESET}")
            except Exception as e:
                print(f"{Colors.RED}Failed to load profile: {e}{Colors.RESET}")
                hardware_profile = None

        # Get custom parameters (with profile-based defaults)
        print(f"\n{Colors.BOLD}{Colors.YELLOW}CUSTOM PARAMETERS:{Colors.RESET}")
        envs_input = get_user_input(f"{Colors.YELLOW}Number of environments (default {default_envs}): {Colors.RESET}")
        envs = envs_input if envs_input else str(default_envs)

        steps_input = get_user_input(f"{Colors.YELLOW}Total timesteps (default {default_steps:,}): {Colors.RESET}")
        steps = steps_input if steps_input else str(default_steps)

        # Display configuration and confirm
        print(f"\n{Colors.BOLD}{Colors.GREEN}TRAINING CONFIGURATION:{Colors.RESET}")
        print(f"{Colors.CYAN}  Market: {market}{Colors.RESET}")
        print(f"{Colors.CYAN}  Parallel Environments: {envs}{Colors.RESET}")
        print(f"{Colors.CYAN}  Total Timesteps: {int(steps):,}{Colors.RESET}")
        if hardware_profile:
            print(f"{Colors.CYAN}  Hardware Profile: {Path(hardware_profile).name}{Colors.RESET}")
        print()

        confirm = prompt_confirm("Start custom JAX training with these settings?", default_yes=True)
        if not confirm:
            print(f"{Colors.YELLOW}Training cancelled.{Colors.RESET}")
            return False

        # Determine which module to use
        script_fixed = self.src_dir / "jax_migration" / "train_ppo_jax_fixed.py"
        script_normal = self.src_dir / "jax_migration" / "train_ppo_jax.py"

        if script_fixed.exists():
            module_name = "src.jax_migration.train_ppo_jax_fixed"
        elif script_normal.exists():
            module_name = "src.jax_migration.train_ppo_jax"
        else:
            print(f"{Colors.RED}JAX training script not found{Colors.RESET}")
            return False

        # Run as module to support relative imports
        command = [
            sys.executable, "-m", module_name,
            "--market", market,
            "--num_envs", envs,
            "--total_timesteps", steps,
            "--data_path", str(self.project_dir / "data" / f"{market}_D1M.csv")
        ]

        success, _ = run_command_with_progress(
            command,
            f"Custom JAX Training ({market}, {envs} envs, {steps} steps)",
            f"jax_custom_{market.lower()}.log"
        )
        return success

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
                clear_screen()
                self.display_banner()
                self.display_main_menu()
                
                choice = get_user_input(
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
                    self.run_stress_test()
                elif choice == "4":
                    self.run_hybrid_test()
                elif choice == "5":
                    self.train_model()
                elif choice == "6":
                    self.run_jax_training_menu()
                elif choice == "7":
                    self.run_evaluation()
                elif choice == "8":
                    print(f"\n{Colors.GREEN}Thank you for using RL TRAINER!{Colors.RESET}")
                    print(f"{Colors.CYAN}Goodbye!{Colors.RESET}")
                    break
                else:
                    print(f"{Colors.RED}Invalid option. Please try again.{Colors.RESET}")
                    time.sleep(2)

                if choice != "8":
                    input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.RESET}")
                    
            except KeyboardInterrupt:
                print(f"\n\n{Colors.YELLOW}Exiting RL TRAINER...{Colors.RESET}")
                break
            except Exception as e:
                error_str = str(e).lower()

                # Check for NumPy version conflict errors
                if any(keyword in error_str for keyword in ['numpy', 'ufunc', '__qualname__', 'ndarray']):
                    print(f"\n{Colors.RED}╔══════════════════════════════════════════════════════════════╗{Colors.RESET}")
                    print(f"{Colors.RED}║          NumPy Version Conflict Detected                      ║{Colors.RESET}")
                    print(f"{Colors.RED}╚══════════════════════════════════════════════════════════════╝{Colors.RESET}")
                    print(f"\n{Colors.YELLOW}This occurs when packages are installed while the program is running.{Colors.RESET}")
                    print(f"{Colors.YELLOW}The program needs to restart to load the new NumPy version.{Colors.RESET}")
                    print(f"\n{Colors.CYAN}Please restart the program:{Colors.RESET}")
                    print(f"{Colors.GREEN}  python main.py{Colors.RESET}")
                    print(f"\n{Colors.YELLOW}Press Enter to exit...{Colors.RESET}")
                    input()
                    sys.exit(0)
                else:
                    # Generic error handling
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
