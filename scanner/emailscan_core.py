#!/usr/bin/env python3
import os
import sys
import json
import argparse
import logging
import re
import time
import threading
from datetime import datetime
from pathlib import Path
import requests
from dotenv import load_dotenv

# Add project root to path to import local modules
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

try:
    from email_phishing_checker import EmailPhishingChecker
    from detector.deepseek_llm_service import DeepSeekLLMService
except ImportError as e:
    print(f"Error: Could not import required modules. {e}")
    sys.exit(1)

# Configure logging (internal file only)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=project_root / 'scanner' / 'scanner.log'
)
logger = logging.getLogger(__name__)

class Spinner:
    def __init__(self, message="Scanning...", delay=0.1):
        self.spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.delay = delay
        self.message = message
        self.running = False
        self.thread = None

    def spin(self):
        idx = 0
        while self.running:
            sys.stdout.write(f"\r  \033[94m{self.spinner[idx % len(self.spinner)]}\033[0m {self.message}")
            sys.stdout.flush()
            idx += 1
            time.sleep(self.delay)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.spin)
        self.thread.start()

    def stop(self, final_message="Done!"):
        self.running = False
        if self.thread:
            self.thread.join()
        sys.stdout.write("\r" + " " * (len(self.message) + 10) + "\r")
        sys.stdout.flush()

class EmailScannerTool:
    def __init__(self, api_key=None, api_url=None):
        # 1. Load from scanner/.env
        load_dotenv(Path(__file__).parent / '.env')
        # 2. Overwrite with project root .env (the actual credentials)
        load_dotenv(project_root / '.env', override=True)
        
        # Determine credentials from .env
        self.default_account = os.getenv('DEFAULT_EMAIL_ACCOUNT', 'gmail').lower()
        if self.default_account == 'gmail':
            self.email = os.getenv('GMAIL_EMAIL')
            self.password = os.getenv('GMAIL_PASSWORD')
            self.provider = 'gmail'
        elif self.default_account == 'outlook':
            self.email = os.getenv('OUTLOOK_EMAIL')
            self.password = os.getenv('OUTLOOK_PASSWORD')
            self.provider = 'outlook'
        else:
            self.email = os.getenv('CUSTOM_EMAIL', os.getenv('EMAIL_USER'))
            self.password = os.getenv('CUSTOM_PASSWORD', os.getenv('EMAIL_PASSWORD'))
            self.provider = 'auto'

        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        self.api_url = api_url or os.getenv('EMAIL_API_URL', 'http://localhost:8000/predict/email/')
        self.max_emails = int(os.getenv('MAX_EMAILS_TO_CHECK', 10))
        
        if not self.api_key:
            self.deepseek_service = None
        else:
            self.deepseek_service = DeepSeekLLMService(self.api_key)
            
        self.ml_checker = EmailPhishingChecker(
            email_address=self.email or "none@none.com", 
            password=self.password or "none",
            provider=self.provider
        )
        self.ml_checker.api_url = self.api_url
        
        # Directory Setup
        self.base_results_dir = Path(__file__).parent / "Scan Results Dir"
        self.day_dir = self.base_results_dir / f"Scan {datetime.now().strftime('%Y-%m-%d')}"
        self.ml_dir = self.day_dir / "ML scans"
        self.ai_dir = self.day_dir / "AI scans"
        self.id_log_file = Path(__file__).parent / ".scanned_ids.json"
        
        for d in [self.ml_dir, self.ai_dir]:
            d.mkdir(parents=True, exist_ok=True)
            
        self.scanned_ids = self._load_scanned_ids()

    def _load_scanned_ids(self):
        if self.id_log_file.exists():
            try:
                with open(self.id_log_file, 'r') as f:
                    return set(json.load(f))
            except:
                return set()
        return set()

    def _save_scanned_ids(self):
        try:
            with open(self.id_log_file, 'w') as f:
                json.dump(list(self.scanned_ids), f)
        except Exception as e:
            logger.error(f"Failed to save scanned IDs: {e}")

    def _sanitize_filename(self, text):
        s = re.sub(r'[^\w\-_\s]', '', text)
        s = s.replace(' ', '_').lower()
        return s[:50]

    def scan(self, subject, body, sender="Unknown", date=None, email_id=None):
        timestamp = datetime.now().strftime("%H%M%S")
        safe_subject = self._sanitize_filename(subject)
        
        email_data = {
            'subject': subject,
            'from': sender,
            'date': date or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'body': body,
            'full_text': f"Subject: {subject}\nFrom: {sender}\n\n{body}",
            'id': email_id or f"cli_{timestamp}"
        }

        print(f"\n\033[1m[SCAN]\033[0m {subject[:60]}...")
        
        # 1. ML Scan
        spinner = Spinner("Performing ML Analysis...")
        spinner.start()
        try:
            ml_result = self.ml_checker.check_for_phishing(email_data)
            verdict = ml_result.get('result', 'UNKNOWN')
            confidence = ml_result.get('confidence_score', 0)
        finally:
            spinner.stop()
        
        sys.stdout.write(f"  > ML Analysis: ")
        if verdict == 'PHISHING':
            print(f"\033[91m{verdict}\033[0m ({confidence}%)")
        else:
            print(f"\033[92m{verdict}\033[0m ({confidence}%)")

        ml_file = self.ml_dir / f"{safe_subject}_ML_{timestamp}.txt"
        with open(ml_file, 'w') as f:
            f.write(f"ML Scan Result - {datetime.now()}\n")
            f.write("="*40 + "\n")
            f.write(f"Subject: {subject}\n")
            f.write(f"From: {sender}\n")
            f.write(f"Result: {verdict}\n")
            f.write(f"Confidence: {confidence}%\n")
            f.write(f"Full Details:\n{json.dumps(ml_result, indent=2)}\n")
        
        # 2. AI Scan
        if self.deepseek_service:
            spinner = Spinner("Performing DeepSeek AI Analysis...")
            spinner.start()
            try:
                ai_analysis = self.deepseek_service.analyze_email_with_report(email_data)
                ai_verdict = ai_analysis.get('verdict', 'UNKNOWN')
            finally:
                spinner.stop()

            sys.stdout.write(f"  > AI Analysis: ")
            if ai_verdict == 'MALICIOUS' or ai_verdict == 'CRITICAL':
                print(f"\033[91m{ai_verdict}\033[0m")
            elif ai_verdict == 'SUSPICIOUS':
                print(f"\033[93m{ai_verdict}\033[0m")
            elif ai_verdict == 'SAFE':
                print(f"\033[92m{ai_verdict}\033[0m")
            else:
                if not ai_analysis.get('api_call_success', True):
                    print(f"\033[91mFAILED\033[0m ({ai_analysis.get('executive_summary', 'API Error')})")
                else:
                    print(f"\033[94m{ai_verdict}\033[0m")

            report_id = f"ai_{timestamp}"
            html_content = self.deepseek_service._generate_html_report(email_data, ai_analysis, report_id)
            
            ai_file = self.ai_dir / f"{safe_subject}_AI_{timestamp}.html"
            with open(ai_file, 'w') as f:
                f.write(html_content)
            
            if email_id:
                self.scanned_ids.add(email_id)
                self._save_scanned_ids()
                
            return ml_result, ai_analysis
        else:
            print(f"  > AI Analysis: \033[93mDISABLED\033[0m (No API Key)")
            if email_id:
                self.scanned_ids.add(email_id)
                self._save_scanned_ids()
            return ml_result, None

    def fetch_and_scan(self, count=None):
        fetch_count = count or self.max_emails
        print(f"\n\033[94m[FETCH]\033[0m Connecting to {self.provider} as {self.email}...")
        
        spinner = Spinner("Accessing secure mail server...")
        spinner.start()
        
        mail = None
        try:
            mail = self.ml_checker.connect_to_email()
            if not mail:
                spinner.stop()
                print("\033[91mERROR: Failed to connect to email server. Check credentials in .env\033[0m")
                return False

            self.ml_checker.max_emails = fetch_count
            emails = self.ml_checker.fetch_recent_emails(mail)
        finally:
            if spinner: spinner.stop()

        if not emails:
            print("\033[93mNo recent emails found to scan.\033[0m")
            return self.show_interactive_menu()

        print(f"\n\033[1mRECENT INBOX STATUS (Last {len(emails)} emails):\033[0m")
        print("━" * 60)
        
        to_scan = []
        for i, email_data in enumerate(emails, 1):
            email_id = email_data.get('id')
            subject = email_data.get('subject', 'No Subject')[:45]
            is_scanned = email_id in self.scanned_ids
            
            status = "\033[96m[SCANNED]\033[0m" if is_scanned else "\033[92m[  NEW  ]\033[0m"
            print(f"  {i:2}. {status} {subject:<45}")
            
            if not is_scanned:
                to_scan.append(email_data)
            time.sleep(0.05) # "Ticking" effect
            
        print("━" * 60)

        if not to_scan:
            print(f"\n\033[92m✓ All {len(emails)} fetched emails have already been secured.\033[0m")
            return self.show_interactive_menu()

        print(f"\n\033[94mStarting analysis for {len(to_scan)} new items...\033[0m")
        
        for email_data in to_scan:
            self.scan(
                email_data.get('subject', 'No Subject'), 
                email_data.get('body', ''), 
                email_data.get('from', 'Unknown'),
                email_data.get('date'),
                email_data.get('id')
            )
            
        print(f"\n\033[92m✓ Scan session complete. {len(to_scan)} new items analyzed.\033[0m")
        return self.show_interactive_menu()

    def show_interactive_menu(self):
        print("\n" + "━" * 40)
        print("\033[1m[MENU]\033[0m What would you like to do?")
        print("  1) \033[94mRescan\033[0m (Fetch again)")
        print("  2) \033[91mQuit\033[0m")
        print("━" * 40)
        
        try:
            choice = input("Select an option (1-2): ").strip()
            if choice == '1':
                return self.fetch_and_scan()
            else:
                print("Goodbye!")
                return False
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            return False

def main():
    parser = argparse.ArgumentParser(description='Email Security Scanner CLI')
    parser.add_argument('--subject', help='Email subject')
    parser.add_argument('--body', help='Email body text')
    parser.add_argument('--file', help='Path to an email text file')
    parser.add_argument('--fetch', type=int, nargs='?', const=0, help='Fetch and scan N recent emails')
    parser.add_argument('--test', action='store_true', help='Run a self-test')

    args = parser.parse_args()
    scanner = EmailScannerTool()

    if args.test:
        scanner.scan("TEST: SECURITY ALERT", "This is a test body.")
        return

    # If --fetch is provided (with or without count) or no args provided
    if args.fetch is not None or (not args.subject and not args.body and not args.file):
        count = args.fetch if (args.fetch and args.fetch > 0) else None
        scanner.fetch_and_scan(count)
        return

    subject = args.subject
    body = args.body

    if args.file:
        file_path = Path(args.file)
        if file_path.exists():
            with open(file_path, 'r') as f:
                content = f.read()
                lines = content.split('\n')
                if not subject: subject = lines[0]
                if not body: body = '\n'.join(lines[1:])
        else:
            print(f"\033[91mERROR: File not found: {args.file}\033[0m")
            sys.exit(1)

    if subject and body:
        scanner.scan(subject, body)
    else:
        print("Usage: ./emailscan [--fetch [N]] [[--subject S --body B] | [--file F]]")

if __name__ == "__main__":
    main()
