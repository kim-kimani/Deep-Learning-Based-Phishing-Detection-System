#!/usr/bin/env python3
"""
Enhanced Email Phishing Checker with DeepSeek Integration
Combines traditional phishing detection with DeepSeek AI analysis and detailed report generation
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deepseek_email_checker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import existing email checker
try:
    from email_phishing_checker import EmailPhishingChecker
except ImportError:
    logger.error("Failed to import EmailPhishingChecker. Make sure email_phishing_checker.py is in the same directory.")
    sys.exit(1)

# Import DeepSeek service
try:
    from detector.deepseek_llm_service import analyze_email_with_deepseek, initialize_deepseek_service
except ImportError:
    logger.error("Failed to import DeepSeek service. Make sure detector/deepseek_llm_service.py exists.")
    sys.exit(1)


class EnhancedEmailChecker:
    """
    Enhanced email checker with DeepSeek AI analysis and report generation
    """
    
    def __init__(self, email_address: str, password: str, 
                 deepseek_api_key: str,
                 provider: str = 'auto', check_interval: int = 300, 
                 max_emails: int = 10, custom_server: str = None,
                 custom_port: int = 993, use_ssl: bool = True):
        """
        Initialize enhanced email checker
        
        Args:
            email_address: Your email address
            password: Your email password
            deepseek_api_key: DeepSeek API key for AI analysis
            provider: Email provider
            check_interval: Check interval in seconds
            max_emails: Maximum emails to check
            custom_server: Custom IMAP server
            custom_port: Custom IMAP port
            use_ssl: Use SSL/TLS
        """
        # Initialize base email checker
        self.base_checker = EmailPhishingChecker(
            email_address=email_address,
            password=password,
            provider=provider,
            check_interval=check_interval,
            max_emails=max_emails,
            custom_server=custom_server,
            custom_port=custom_port,
            use_ssl=use_ssl
        )
        
        # Store DeepSeek API key
        self.deepseek_api_key = deepseek_api_key
        
        # Initialize DeepSeek service
        if not initialize_deepseek_service(deepseek_api_key):
            logger.warning("DeepSeek service initialization failed. AI analysis will be disabled.")
            self.deepseek_enabled = False
        else:
            self.deepseek_enabled = True
            logger.info("DeepSeek AI analysis enabled")
        
        # Report directory
        self.reports_dir = Path("enhanced_email_reports")
        self.reports_dir.mkdir(exist_ok=True)
        
        # Statistics
        self.stats = {
            "total_emails_checked": 0,
            "phishing_detected": 0,
            "deepseek_analyses": 0,
            "reports_generated": 0,
            "start_time": datetime.now().isoformat()
        }
    
    def check_email_with_deepseek(self, email_data: Dict) -> Dict:
        """
        Check email with both traditional phishing detection and DeepSeek AI analysis
        
        Args:
            email_data: Email data from base checker
            
        Returns:
            Dict: Combined analysis results
        """
        result = {
            "email_metadata": {
                "subject": email_data.get('subject', 'No Subject'),
                "from": email_data.get('from', 'Unknown'),
                "date": email_data.get('date', 'Unknown'),
                "email_id": email_data.get('id', 'Unknown')
            },
            "traditional_analysis": None,
            "deepseek_analysis": None,
            "combined_verdict": "UNKNOWN",
            "report_generated": False,
            "report_files": None
        }
        
        # Step 1: Traditional phishing analysis
        try:
            traditional_result = self.base_checker.check_for_phishing(email_data)
            result["traditional_analysis"] = traditional_result
            
            # Update statistics
            self.stats["total_emails_checked"] += 1
            if traditional_result.get('result') == 'PHISHING':
                self.stats["phishing_detected"] += 1
            
        except Exception as e:
            logger.error(f"Traditional analysis failed: {e}")
            result["traditional_analysis"] = {
                "success": False,
                "error": str(e),
                "result": "ERROR"
            }
        
        # Step 2: DeepSeek AI analysis (if enabled)
        if self.deepseek_enabled:
            try:
                # Use the UUID from Django if available
                if result["traditional_analysis"] and result["traditional_analysis"].get("prediction_id"):
                    django_uuid = result["traditional_analysis"]["prediction_id"]
                    email_data['id'] = django_uuid
                    logger.info(f"Updated email ID to Django UUID: {django_uuid}")

                logger.info(f"Running DeepSeek analysis for: {email_data.get('subject', 'Unknown')}")
                
                deepseek_result = analyze_email_with_deepseek(email_data, self.deepseek_api_key)
                result["deepseek_analysis"] = deepseek_result
                
                if deepseek_result.get("success", False):
                    self.stats["deepseek_analyses"] += 1
                    
                    # Check if report was generated
                    if deepseek_result.get("analysis", {}).get("report_generation", {}).get("success", False):
                        self.stats["reports_generated"] += 1
                        result["report_generated"] = True
                        result["report_files"] = deepseek_result["analysis"]["report_generation"]["report_files"]
                        
                        logger.info(f"DeepSeek report generated: {result['report_files']}")
                
                # Determine combined verdict
                result["combined_verdict"] = self._determine_combined_verdict(
                    result["traditional_analysis"],
                    result["deepseek_analysis"]
                )
                
            except Exception as e:
                logger.error(f"DeepSeek analysis failed: {e}")
                result["deepseek_analysis"] = {
                    "error": True,
                    "error_message": str(e),
                    "success": False
                }
        
        return result
    
    def _determine_combined_verdict(self, traditional: Dict, deepseek: Dict) -> str:
        """
        Determine combined verdict based on both analyses
        
        Args:
            traditional: Traditional analysis results
            deepseek: DeepSeek analysis results
            
        Returns:
            str: Combined verdict
        """
        # Default to traditional analysis
        if not traditional.get("success", False):
            return "ERROR"
        
        traditional_result = traditional.get("result", "UNKNOWN")
        
        # If DeepSeek analysis is available, use it to enhance decision
        if deepseek and deepseek.get("success", False):
            deepseek_analysis = deepseek.get("analysis", {})
            deepseek_verdict = deepseek_analysis.get("verdict", "UNKNOWN")
            
            # Map DeepSeek verdict to our system
            verdict_mapping = {
                "SAFE": "NOT PHISHING",
                "SUSPICIOUS": "SUSPICIOUS",
                "MALICIOUS": "PHISHING",
                "CRITICAL": "PHISHING"
            }
            
            deepseek_mapped = verdict_mapping.get(deepseek_verdict, "UNKNOWN")
            
            # If both agree, use that verdict
            if traditional_result == deepseek_mapped:
                return traditional_result
            
            # If DeepSeek says PHISHING but traditional says NOT PHISHING, mark as SUSPICIOUS
            if deepseek_mapped == "PHISHING" and traditional_result == "NOT PHISHING":
                return "SUSPICIOUS"
            
            # If traditional says PHISHING but DeepSeek says SAFE, still mark as SUSPICIOUS
            if traditional_result == "PHISHING" and deepseek_mapped == "NOT PHISHING":
                return "SUSPICIOUS"
            
            # Default to traditional result
            return traditional_result
        
        # Only traditional analysis available
        return traditional_result
    
    def process_emails_with_deepseek(self, mail) -> List[Dict]:
        """
        Process emails with DeepSeek analysis
        
        Args:
            mail: IMAP connection
            
        Returns:
            List[Dict]: Analysis results for all emails
        """
        results = []
        
        # Fetch recent emails
        emails = self.base_checker.fetch_recent_emails(mail)
        
        logger.info(f"Processing {len(emails)} emails with DeepSeek analysis...")
        
        for i, email_data in enumerate(emails):
            logger.info(f"Processing email {i+1}/{len(emails)}: {email_data['subject'][:50]}...")
            
            # Check email with DeepSeek
            result = self.check_email_with_deepseek(email_data)
            results.append(result)
            
            # Log results
            self._log_email_result(result)
            
            # Small delay to avoid rate limiting
            time.sleep(1)
        
        return results
    
    def _log_email_result(self, result: Dict):
        """Log email analysis results"""
        subject = result["email_metadata"]["subject"]
        traditional_result = result.get("traditional_analysis", {}).get("result", "UNKNOWN")
        deepseek_verdict = result.get("deepseek_analysis", {}).get("analysis", {}).get("verdict", "UNKNOWN")
        combined_verdict = result.get("combined_verdict", "UNKNOWN")
        
        log_message = f"Email: {subject[:50]}... | Traditional: {traditional_result} | DeepSeek: {deepseek_verdict} | Combined: {combined_verdict}"
        
        if combined_verdict == "PHISHING":
            logger.warning(f"⚠️  {log_message}")
        elif combined_verdict == "SUSPICIOUS":
            logger.info(f"🔍 {log_message}")
        else:
            logger.info(f"✅ {log_message}")
    
    def run_single_check(self):
        """Run a single enhanced email check"""
        logger.info("Running enhanced email check with DeepSeek analysis...")
        
        # Connect to email server
        mail = self.base_checker.connect_to_email()
        if not mail:
            logger.error("Failed to connect to email server")
            return
        
        try:
            # Process emails with DeepSeek
            results = self.process_emails_with_deepseek(mail)
            
            # Print summary
            self._print_summary(results)
            
            # Save results
            self._save_results(results)
            
            # Print statistics
            self._print_statistics()
            
        finally:
            try:
                mail.close()
                mail.logout()
            except:
                pass
    
    def _print_summary(self, results: List[Dict]):
        """Print analysis summary"""
        print("\n" + "="*80)
        print("ENHANCED EMAIL SECURITY ANALYSIS WITH DEEPSEEK AI")
        print("="*80)
        
        phishing_count = sum(1 for r in results if r.get("combined_verdict") == "PHISHING")
        suspicious_count = sum(1 for r in results if r.get("combined_verdict") == "SUSPICIOUS")
        safe_count = sum(1 for r in results if r.get("combined_verdict") == "NOT PHISHING")
        
        print(f"\n📊 ANALYSIS SUMMARY:")
        print(f"   📧 Total emails analyzed: {len(results)}")
        print(f"   ⚠️  Phishing detected: {phishing_count}")
        print(f"   🔍 Suspicious emails: {suspicious_count}")
        print(f"   ✅ Safe emails: {safe_count}")
        print(f"   🤖 DeepSeek analyses: {self.stats['deepseek_analyses']}")
        print(f"   📄 Reports generated: {self.stats['reports_generated']}")
        
        # List phishing emails
        phishing_emails = [r for r in results if r.get("combined_verdict") == "PHISHING"]
        if phishing_emails:
            print(f"\n🔴 PHISHING EMAILS DETECTED:")
            for i, email_result in enumerate(phishing_emails, 1):
                metadata = email_result["email_metadata"]
                traditional = email_result.get("traditional_analysis", {})
                deepseek = email_result.get("deepseek_analysis", {}).get("analysis", {})
                
                print(f"\n  {i}. {metadata['subject'][:60]}...")
                print(f"     From: {metadata['from']}")
                print(f"     Traditional confidence: {traditional.get('confidence_score', 0)}%")
                
                if deepseek:
                    print(f"     DeepSeek threat level: {deepseek.get('threat_assessment', {}).get('overall_threat_level', 'UNKNOWN')}")
                    print(f"     DeepSeek risk score: {deepseek.get('risk_scoring', {}).get('overall_risk_score', 0)}/100")
                    
                    # Check if report was generated
                    if email_result.get("report_generated", False):
                        report_files = email_result.get("report_files", {})
                        if report_files.get("html"):
                            print(f"     📄 Report: {report_files['html']}")
        
        # List suspicious emails
        suspicious_emails = [r for r in results if r.get("combined_verdict") == "SUSPICIOUS"]
        if suspicious_emails:
            print(f"\n🟡 SUSPICIOUS EMAILS (Require Review):")
            for i, email_result in enumerate(suspicious_emails, 1):
                metadata = email_result["email_metadata"]
                print(f"  {i}. {metadata['subject'][:60]}...")
                print(f"     From: {metadata['from']}")
                
                if email_result.get("report_generated", False):
                    report_files = email_result.get("report_files", {})
                    if report_files.get("html"):
                        print(f"     📄 Detailed report available: {report_files['html']}")
    
    def _save_results(self, results: List[Dict]):
        """Save analysis results to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.reports_dir / f"enhanced_analysis_{timestamp}.json"
            
            # Prepare data for saving
            save_data = {
                "analysis_timestamp": datetime.now().isoformat(),
                "statistics": self.stats,
                "email_results": [],
                "summary": {
                    "total_emails": len(results),
                    "phishing_count": sum(1 for r in results if r.get("combined_verdict") == "PHISHING"),
                    "suspicious_count": sum(1 for r in results if r.get("combined_verdict") == "SUSPICIOUS"),
                    "safe_count": sum(1 for r in results if r.get("combined_verdict") == "NOT PHISHING")
                }
            }
            
            # Add email results (simplified for storage)
            for result in results:
                simplified_result = {
                    "subject": result["email_metadata"]["subject"],
                    "from": result["email_metadata"]["from"],
                    "traditional_result": result.get("traditional_analysis", {}).get("result", "UNKNOWN"),
                    "traditional_confidence": result.get("traditional_analysis", {}).get("confidence_score", 0),
                    "deepseek_verdict": result.get("deepseek_analysis", {}).get("analysis", {}).get("verdict", "UNKNOWN"),
                    "combined_verdict": result.get("combined_verdict", "UNKNOWN"),
                    "report_generated": result.get("report_generated", False),
                    "report_files": result.get("report_files")
                }
                save_data["email_results"].append(simplified_result)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Enhanced analysis results saved to: {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def _print_statistics(self):
        """Print system statistics"""
        print("\n" + "="*80)
        print("SYSTEM STATISTICS")
        print("="*80)
        
        runtime = datetime.now() - datetime.fromisoformat(self.stats["start_time"])
        runtime_seconds = runtime.total_seconds()
        
        print(f"\n⏱️  Runtime: {runtime_seconds:.1f} seconds")
        print(f"📧 Total emails checked: {self.stats['total_emails_checked']}")
        print(f"⚠️  Phishing detected: {self.stats['phishing_detected']}")
        print(f"🤖 DeepSeek analyses: {self.stats['deepseek_analyses']}")
        print(f"📄 Reports generated: {self.stats['reports_generated']}")
        
        if self.stats['total_emails_checked'] > 0:
            phishing_rate = (self.stats['phishing_detected'] / self.stats['total_emails_checked']) * 100
            print(f"📊 Phishing rate: {phishing_rate:.1f}%")
        
        print(f"\n📁 Report directory: {self.reports_dir.absolute()}")
        print("   - HTML reports: Open in browser")
        print("   - JSON reports: Machine-readable data")
        print("   - Text reports: Quick review")


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Enhanced Email Checker with DeepSeek AI Analysis')
    
    # Email credentials
    parser.add_argument('--email', required=True, help='Your email address')
    parser.add_argument('--password', required=True, help='Your email password')
    
    # DeepSeek API key
    parser.add_argument('--deepseek-key', required=True, help='DeepSeek API key for AI analysis')
    
    # Email provider options
    parser.add_argument('--provider', default='auto', 
                       help='Email provider (gmail, outlook, yahoo, etc.) or "auto" to detect')
    parser.add_argument('--max-emails', type=int, default=5,
                       help='Maximum emails to analyze (default: 5)')
    parser.add_argument('--server', help='Custom IMAP server (for custom provider)')
    parser.add_argument('--port', type=int, default=993, help='Custom IMAP port (default: 993)')
    parser.add_argument('--no-ssl', action='store_true', help='Disable SSL/TLS (not recommended)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("ENHANCED EMAIL CHECKER WITH DEEPSEEK AI ANALYSIS")
    print("="*80)
    print(f"Email: {args.email}")
    print(f"Provider: {args.provider}")
    print(f"Max emails: {args.max_emails}")
    print(f"DeepSeek AI: Enabled")
    print("="*80)
    
    # Check if Django server is running
    try:
        import requests
        response = requests.get('http://localhost:8000/', timeout=5)
        if response.status_code != 200:
            logger.warning("Django server might not be running properly")
    except:
        logger.error("❌ Django server is not running!")
        logger.error("Please start it first: python manage.py runserver")
        sys.exit(1)
    
    # Create enhanced checker
    checker = EnhancedEmailChecker(
        email_address=args.email,
        password=args.password,
        deepseek_api_key=args.deepseek_key,
        provider=args.provider,
        max_emails=args.max_emails,
        custom_server=args.server,
        custom_port=args.port,
        use_ssl=not args.no_ssl
    )
    
    # Run single check
    checker.run_single_check()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\n📁 Generated reports are in: enhanced_email_reports/")
    print("📊 View HTML reports in your browser")
    print("📝 Check logs: deepseek_email_checker.log")
    print("\nTo run again with different settings:")
    print(f"  python email_checker_with_deepseek.py --email {args.email} --password [PASSWORD] --deepseek-key [API_KEY]")


if __name__ == "__main__":
    main()
