#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys


# !/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys


def main():
    """Run administrative tasks."""
    # Fix Windows console encoding for Unicode characters
    if sys.platform.startswith('win'):
        # Set UTF-8 encoding for Windows console
        try:
            os.system('chcp 65001 > nul')
        except Exception:
            pass
        
        # Force UTF-8 encoding in Python
        if hasattr(sys.stdout, 'reconfigure'):
            try:
                sys.stdout.reconfigure(encoding='utf-8')
                sys.stderr.reconfigure(encoding='utf-8')
            except Exception:
                pass
        
        # Set environment variables
        os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    os.environ.setdefault(
        'DJANGO_SETTINGS_MODULE',
        'ml_security_detector.settings'
    )
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
