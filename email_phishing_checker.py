#!/usr/bin/env python3
"""
Universal Email Phishing Checker
Works with any IMAP-enabled email provider (Gmail, Outlook, Yahoo, etc.)
"""

import imaplib
import email
import email.policy
import json
import time
import requests
from datetime import datetime, timedelta
import os
import sys
from typing import List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('email_checker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EmailPhishingChecker:
    """Universal email checker for any IMAP provider"""
    
    # Common email provider IMAP settings
    PROVIDER_SETTINGS = {
        'gmail': {
            'server': 'imap.gmail.com',
            'port': 993,
            'ssl': True,
            'requires_app_password': True
        },
        'outlook': {
            'server': 'outlook.office365.com',
            'port': 993,
            'ssl': True,
            'requires_app_password': False
        },
        'yahoo': {
            'server': 'imap.mail.yahoo.com',
            'port': 993,
            'ssl': True,
            'requires_app_password': False
        },
        'icloud': {
            'server': 'imap.mail.me.com',
            'port': 993,
            'ssl': True,
            'requires_app_password': True
        },
        'aol': {
            'server': 'imap.aol.com',
            'port': 993,
            'ssl': True,
            'requires_app_password': False
        },
        'zoho': {
            'server': 'imap.zoho.com',
            'port': 993,
            'ssl': True,
            'requires_app_password': False
        },
        'custom': {
            'server': '',
            'port': 993,
            'ssl': True,
            'requires_app_password': False
        }
    }
    
    def __init__(self, email_address: str, password: str, 
                 provider: str = 'auto', check_interval: int = 300, 
                 max_emails: int = 10, custom_server: str = None,
                 custom_port: int = 993, use_ssl: bool = True):
        """
        Initialize email checker
        
        Args:
            email_address: Your email address
            password: Your email password or app password
            provider: Email provider ('gmail', 'outlook', 'yahoo', etc.) or 'auto' to detect
            check_interval: How often to check emails in seconds
            max_emails: Maximum number of emails to check per run
            custom_server: Custom IMAP server (for 'custom' provider)
            custom_port: Custom IMAP port
            use_ssl: Use SSL/TLS connection
        """
        self.email_address = email_address
        self.password = password
        self.check_interval = check_interval
        self.max_emails = max_emails
        self.api_url = "http://localhost:8000/predict/email/"
        
        # Determine provider settings
        self.provider_config = self._get_provider_config(provider, custom_server, custom_port, use_ssl)
        
        # Track processed emails to avoid duplicates
        self.processed_uids = set()
        
        logger.info(f"Initialized email checker for {email_address}")
        logger.info(f"Provider: {provider} ({self.provider_config['server']}:{self.provider_config['port']})")
        logger.info(f"Check interval: {check_interval} seconds")
        logger.info(f"Phishing API: {self.api_url}")
    
    def _get_provider_config(self, provider: str, custom_server: str, 
                            custom_port: int, use_ssl: bool) -> Dict:
        """Get IMAP configuration for the specified provider"""
        if provider == 'auto':
            # Auto-detect provider from email domain
            domain = self.email_address.split('@')[-1].lower()
            
            if 'gmail.com' in domain:
                provider = 'gmail'
            elif 'outlook.com' in domain or 'hotmail.com' in domain or 'live.com' in domain:
                provider = 'outlook'
            elif 'yahoo.com' in domain:
                provider = 'yahoo'
            elif 'icloud.com' in domain or 'me.com' in domain:
                provider = 'icloud'
            elif 'aol.com' in domain:
                provider = 'aol'
            elif 'zoho.com' in domain:
                provider = 'zoho'
            else:
                provider = 'custom'
                logger.info(f"Auto-detected custom provider for domain: {domain}")
        
        if provider not in self.PROVIDER_SETTINGS:
            logger.warning(f"Unknown provider '{provider}', using custom settings")
            provider = 'custom'
        
        config = self.PROVIDER_SETTINGS[provider].copy()
        
        # Override with custom settings if provided
        if custom_server:
            config['server'] = custom_server
        if custom_port:
            config['port'] = custom_port
        config['ssl'] = use_ssl
        
        return config
    
    def connect_to_email(self) -> Optional[imaplib.IMAP4_SSL]:
        """Connect to email server using IMAP"""
        try:
            server = self.provider_config['server']
            port = self.provider_config['port']
            use_ssl = self.provider_config['ssl']
            
            if not server:
                logger.error("No IMAP server specified")
                return None
            
            logger.info(f"Connecting to {server}:{port} (SSL: {use_ssl})...")
            
            if use_ssl:
                mail = imaplib.IMAP4_SSL(server, port)
            else:
                mail = imaplib.IMAP4(server, port)
                # Start TLS if supported
                try:
                    mail.starttls()
                except:
                    logger.warning("TLS not supported by server")
            
            # Login
            mail.login(self.email_address, self.password)
            
            # Select inbox
            mail.select('inbox')
            
            logger.info("Successfully connected to email server")
            return mail
            
        except imaplib.IMAP4.error as e:
            logger.error(f"Authentication failed: {e}")
            
            # Provide helpful error messages
            if self.provider_config.get('requires_app_password', False):
                logger.info("This provider requires an app password (not your regular password):")
                logger.info("  Gmail: Enable 2FA, then generate app password")
                logger.info("  iCloud: Generate app-specific password at appleid.apple.com")
            else:
                logger.info("Check your username and password")
                
            return None
        except Exception as e:
            logger.error(f"Failed to connect to email server: {e}")
            return None
    
    def fetch_recent_emails(self, mail: imaplib.IMAP4_SSL, hours_back: int = 24) -> List[Dict]:
        """
        Fetch recent emails from email server
        
        Args:
            mail: IMAP connection
            hours_back: How many hours back to check
            
        Returns:
            List of email dictionaries with content
        """
        emails = []
        
        try:
            # Calculate search date
            date_since = (datetime.now() - timedelta(hours=hours_back)).strftime("%d-%b-%Y")
            
            # Search for recent emails
            result, data = mail.search(None, f'(SINCE "{date_since}")')
            
            if result != 'OK':
                logger.error("Failed to search emails")
                return emails
            
            # Get list of email IDs
            email_ids = data[0].split()
            
            # Limit to max_emails, starting from most recent
            email_ids = email_ids[-self.max_emails:] if len(email_ids) > self.max_emails else email_ids
            
            logger.info(f"Found {len(email_ids)} recent emails")
            
            for i, email_id in enumerate(email_ids):
                try:
                    # Skip if already processed
                    if email_id in self.processed_uids:
                        continue
                    
                    # Fetch the email
                    result, msg_data = mail.fetch(email_id, '(RFC822)')
                    
                    if result != 'OK':
                        logger.warning(f"Failed to fetch email {email_id}")
                        continue
                    
                    # Parse email
                    raw_email = msg_data[0][1]
                    msg = email.message_from_bytes(raw_email, policy=email.policy.default)
                    
                    # Extract email content
                    email_info = self._parse_email(msg, email_id)
                    
                    if email_info:
                        emails.append(email_info)
                        self.processed_uids.add(email_id)
                        
                        logger.debug(f"Processed email {i+1}/{len(email_ids)}: {email_info.get('subject', 'No subject')}")
                        
                except Exception as e:
                    logger.error(f"Error processing email {email_id}: {e}")
                    continue
            
            logger.info(f"Successfully parsed {len(emails)} emails")
            return emails
            
        except Exception as e:
            logger.error(f"Error fetching emails: {e}")
            return emails
    
    def _parse_email(self, msg: email.message.EmailMessage, email_id: bytes) -> Optional[Dict]:
        """Parse email message into structured data"""
        try:
            # Extract basic headers
            subject = msg.get('subject', 'No Subject')
            from_addr = msg.get('from', 'Unknown Sender')
            date = msg.get('date', 'Unknown Date')
            
            # Extract email body
            body = self._extract_email_body(msg)
            
            if not body:
                logger.warning(f"Email {email_id} has no readable body")
                return None
            
            # Combine all text for analysis
            full_text = f"Subject: {subject}\nFrom: {from_addr}\nDate: {date}\n\n{body}"
            
            return {
                'id': email_id.decode(),
                'subject': subject,
                'from': from_addr,
                'date': date,
                'body': body,
                'full_text': full_text,
                'original_msg': msg
            }
            
        except Exception as e:
            logger.error(f"Error parsing email: {e}")
            return None
    
    def _extract_email_body(self, msg: email.message.EmailMessage) -> str:
        """Extract text body from email message"""
        body_parts = []
        
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))
                
                # Skip attachments
                if "attachment" in content_disposition:
                    continue
                
                # Get text content
                if content_type == "text/plain":
                    try:
                        body = part.get_content()
                        if body:
                            body_parts.append(body)
                    except:
                        pass
                elif content_type == "text/html":
                    # Simple HTML to text conversion
                    try:
                        import html2text
                        h = html2text.HTML2Text()
                        h.ignore_links = False
                        h.ignore_images = True
                        html_content = part.get_content()
                        text_content = h.handle(html_content)
                        body_parts.append(text_content)
                    except:
                        # Fallback: just get the raw content
                        try:
                            body = part.get_content()
                            if body:
                                body_parts.append(body)
                        except:
                            pass
        else:
            # Not multipart, get the content directly
            content_type = msg.get_content_type()
            if content_type in ["text/plain", "text/html"]:
                try:
                    body = msg.get_content()
                    if body:
                        body_parts.append(body)
                except:
                    pass
        
        # Combine all body parts
        return "\n\n".join(body_parts) if body_parts else ""
    
    def check_for_phishing(self, email_data: Dict) -> Dict:
        """
        Check if an email is phishing using the detection API
        
        Args:
            email_data: Email dictionary with full_text
            
        Returns:
            Dictionary with phishing analysis results
        """
        try:
            # Prepare request to phishing detection API
            payload = {
                "email_text": email_data['full_text']
            }
            
            # Make API request
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Add email metadata to result
                result['email_metadata'] = {
                    'subject': email_data['subject'],
                    'from': email_data['from'],
                    'date': email_data['date'],
                    'email_id': email_data['id']
                }
                
                return result
            else:
                logger.error(f"API request failed: {response.status_code}")
                return {
                    'success': False,
                    'error': f"API returned status {response.status_code}",
                    'email_metadata': {
                        'subject': email_data['subject'],
                        'from': email_data['from']
                    }
                }
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error checking email: {e}")
            return {
                'success': False,
                'error': str(e),
                'email_metadata': {
                    'subject': email_data['subject'],
                    'from': email_data['from']
                }
            }
        except Exception as e:
            logger.error(f"Error checking email: {e}")
            return {
                'success': False,
                'error': str(e),
                'email_metadata': {
                    'subject': email_data['subject'],
                    'from': email_data['from']
                }
            }
    
    def process_emails(self, mail: imaplib.IMAP4_SSL) -> List[Dict]:
        """
        Process all recent emails and check for phishing
        
        Args:
            mail: IMAP connection
            
        Returns:
            List of phishing check results
        """
        results = []
        
        # Fetch recent emails
        emails = self.fetch_recent_emails(mail)
        
        logger.info(f"Checking {len(emails)} emails for phishing...")
        
        # Check each email
        for i, email_data in enumerate(emails):
            logger.info(f"Checking email {i+1}/{len(emails)}: {email_data['subject'][:50]}...")
            
            result = self.check_for_phishing(email_data)
            results.append(result)
            
            # Log phishing detection
            if result.get('success') and result.get('result') == 'PHISHING':
                logger.warning(f"⚠️ PHISHING DETECTED: {email_data['subject']}")
                logger.warning(f"   From: {email_data['from']}")
                logger.warning(f"   Confidence: {result.get('confidence_score', 0)}%")
            
            # Small delay to avoid overwhelming the API
            time.sleep(0.5)
        
        return results
    
    def run_continuous_check(self):
        """Run continuous email checking at specified interval"""
        logger.info("Starting continuous email checking...")
        logger.info(f"Will check every {self.check_interval} seconds")
        logger.info("Press Ctrl+C to stop")
        
        try:
            while True:
                logger.info("-" * 50)
                logger.info(f"Checking emails at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Connect to email server
                mail = self.connect_to_email()
                if not mail:
                    logger.error("Failed to connect to email server, retrying in 60 seconds")
                    time.sleep(60)
                    continue
                
                try:
                    # Process emails
                    results = self.process_emails(mail)
                    
                    # Log summary
                    phishing_count = sum(1 for r in results if r.get('result') == 'PHISHING')
                    logger.info(f"Check complete: {len(results)} emails processed, {phishing_count} phishing detected")
                    
                    # Save results to file
                    self._save_results(results)
                    
                finally:
                    # Close connection
                    try:
                        mail.close()
                        mail.logout()
                    except:
                        pass
                
                # Wait for next check
                logger.info(f"Next check in {self.check_interval} seconds...")
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            logger.info("\nStopped by user")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
    
    def _save_results(self, results: List[Dict]):
        """Save phishing check results to JSON file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"phishing_results_{timestamp}.json"
            
            # Filter only successful results
            valid_results = []
            for result in results:
                if result.get('success'):
                    # Simplify the result for storage
                    simple_result = {
                        'timestamp': datetime.now().isoformat(),
                        'subject': result.get('email_metadata', {}).get('subject', 'Unknown'),
                        'from': result.get('email_metadata', {}).get('from', 'Unknown'),
                        'result': result.get('result', 'UNKNOWN'),
                        'confidence_score': result.get('confidence_score', 0),
                        'confidence_level': result.get('confidence_level', 'Unknown'),
                        'threat_detected': result.get('enhanced_analysis', {}).get('threat_detected', False)
                    }
                    valid_results.append(simple_result)
            
            if valid_results:
                with open(filename, 'w') as f:
                    json.dump(valid_results, f, indent=2)
                logger.info(f"Results saved to {filename}")
                
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def run_single_check(self):
        """Run a single email check and print results"""
        logger.info("Running single email check...")
        
        mail = self.connect_to_email()
        if not mail:
            logger.error("Failed to connect to email server")
            return
        
        try:
            results = self.process_emails(mail)
            
            # Print summary
            print("\n" + "="*60)
            print("PHISHING EMAIL CHECK RESULTS")
            print("="*60)
            
            phishing_emails = [r for r in results if r.get('result') == 'PHISHING']
            safe_emails = [r for r in results if r.get('result') == 'NOT PHISHING']
            
            print(f"\n📧 Total emails checked: {len(results)}")
            print(f"⚠️  Phishing detected: {len(phishing_emails)}")
            print(f"✅ Safe emails: {len(safe_emails)}")
            
            if phishing_emails:
                print("\n🔴 PHISHING EMAILS FOUND:")
                for i, email_result in enumerate(phishing_emails, 1):
                    metadata = email_result.get('email_metadata', {})
                    print(f"\n  {i}. {metadata.get('subject', 'No subject')}")
                    print(f"     From: {metadata.get('from', 'Unknown')}")
                    print(f"     Confidence: {email_result.get('confidence_score', 0)}%")
                    print(f"     Level: {email_result.get('confidence_level', 'Unknown')}")
            
            # Save results
            self._save_results(results)
            
        finally:
            try:
                mail.close()
                mail.logout()
            except:
                pass


def main():
    """Main function with command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Universal Email Phishing Checker')
    parser.add_argument('--email', required=True, help='Your email address')
    parser.add_argument('--password', required=True, help='Your email password')
    parser.add_argument('--provider', default='auto', 
                       help='Email provider (gmail, outlook, yahoo, icloud, aol, zoho, custom) or "auto" to detect')
    parser.add_argument('--interval', type=int, default=300, 
                       help='Check interval in seconds (default: 300)')
    parser.add_argument('--max-emails', type=int, default=10,
                       help='Maximum emails to check per run (default: 10)')
    parser.add_argument('--continuous', action='store_true',
                       help='Run continuously at specified interval')
    parser.add_argument('--single', action='store_true',
                       help='Run a single check and exit')
    parser.add_argument('--api-url', default='http://localhost:8000/predict/email/',
                       help='Phishing detection API URL')
    parser.add_argument('--server', help='Custom IMAP server (for custom provider)')
    parser.add_argument('--port', type=int, default=993, help='Custom IMAP port (default: 993)')
    parser.add_argument('--no-ssl', action='store_true', help='Disable SSL/TLS (not recommended)')
    
    args = parser.parse_args()
    
    # Check if Django server is running
    try:
        response = requests.get('http://localhost:8000/', timeout=5)
        if response.status_code != 200:
            logger.warning("Django server might not be running properly")
    except:
        logger.error("❌ Django server is not running!")
        logger.error("Please start it first: python manage.py runserver")
        sys.exit(1)
    
    # Create checker instance
    checker = EmailPhishingChecker(
        email_address=args.email,
        password=args.password,
        provider=args.provider,
        check_interval=args.interval,
        max_emails=args.max_emails,
        custom_server=args.server,
        custom_port=args.port,
        use_ssl=not args.no_ssl
    )
    
    # Override API URL if provided
    if args.api_url:
        checker.api_url = args.api_url
    
    # Run based on mode
    if args.continuous:
        checker.run_continuous_check()
    elif args.single:
        checker.run_single_check()
    else:
        # Default: single check
        checker.run_single_check()


if __name__ == "__main__":
    # Check if required libraries are available
    try:
        import html2text
    except ImportError:
        logger.warning("html2text not installed. HTML emails may not be parsed correctly.")
        logger.info("Install it with: pip install html2text")
    
    main()
