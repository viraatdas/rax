#!/usr/bin/env python3
"""RAX CLI - Command-line interface for the RAX safe JAX execution layer."""

import argparse
import sys
from pathlib import Path

from . import __version__
from .runner import run_script


def main(argv=None):
    """Main CLI entry point for RAX."""
    parser = argparse.ArgumentParser(
        prog='rax',
        description='RAX: A safer, compiler-like JAX frontend',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  rax run script.py             # Run a JAX script with safety checks
  rax run script.py -- --arg1   # Pass arguments to the script
  
Future commands:
  rax compile script.py         # Validate without execution
  rax trace script.py           # Output compiled JAXPR
  rax lint script.py            # Detect missing annotations
  rax export script.py          # Export to XLA/MLIR
"""
    )
    
    parser.add_argument(
        '-v', '--version',
        action='version',
        version=f'%(prog)s {__version__}'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # 'run' command
    run_parser = subparsers.add_parser(
        'run',
        help='Run a JAX script with compile-time safety checks'
    )
    run_parser.add_argument(
        'script',
        type=str,
        help='Path to the Python script to run'
    )
    run_parser.add_argument(
        'script_args',
        nargs=argparse.REMAINDER,
        help='Arguments to pass to the script'
    )
    run_parser.add_argument(
        '--no-jit',
        action='store_true',
        help='Disable automatic JIT compilation (not recommended)'
    )
    run_parser.add_argument(
        '--no-monkeypatch',
        action='store_true',
        help='Disable JAX operation monkeypatching'
    )
    run_parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    # Future commands placeholder
    compile_parser = subparsers.add_parser(
        'compile',
        help='Validate script without execution (coming soon)'
    )
    compile_parser.add_argument('script', type=str, help='Script to compile')
    
    trace_parser = subparsers.add_parser(
        'trace',
        help='Output compiled JAXPR (coming soon)'
    )
    trace_parser.add_argument('script', type=str, help='Script to trace')
    
    lint_parser = subparsers.add_parser(
        'lint',
        help='Detect missing annotations (coming soon)'
    )
    lint_parser.add_argument('script', type=str, help='Script to lint')
    
    export_parser = subparsers.add_parser(
        'export',
        help='Export to XLA/MLIR (coming soon)'
    )
    export_parser.add_argument('script', type=str, help='Script to export')
    
    args = parser.parse_args(argv)
    
    if not args.command:
        parser.print_help()
        return 1
    
    if args.command == 'run':
        script_path = Path(args.script)
        if not script_path.exists():
            print(f"Error: Script '{args.script}' not found.", file=sys.stderr)
            return 1
        
        try:
            # Update sys.argv for the script
            sys.argv = [args.script] + args.script_args
            
            # Run the script with RAX safety
            run_script(
                str(script_path),
                enable_jit=not args.no_jit,
                enable_monkeypatch=not args.no_monkeypatch,
                verbose=args.verbose
            )
            return 0
        except Exception as e:
            print(f"\n[RAX] Error: {e}", file=sys.stderr)
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1
    
    else:
        print(f"Command '{args.command}' is not yet implemented.", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main()) 