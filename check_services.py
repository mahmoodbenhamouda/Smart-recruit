"""
Quick system status checker
Run this to verify all services are up and running
"""

import requests
import sys
from colorama import init, Fore, Style

init(autoreset=True)

def check_service(name, url, expected_status=200):
    """Check if a service is responding"""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == expected_status:
            print(f"{Fore.GREEN}✅ {name}: RUNNING{Style.RESET_ALL}")
            return True
        else:
            print(f"{Fore.RED}❌ {name}: UNHEALTHY (status {response.status_code}){Style.RESET_ALL}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"{Fore.RED}❌ {name}: NOT RUNNING (connection refused){Style.RESET_ALL}")
        return False
    except requests.exceptions.Timeout:
        print(f"{Fore.YELLOW}⚠️  {name}: TIMEOUT (might be slow){Style.RESET_ALL}")
        return False
    except Exception as e:
        print(f"{Fore.RED}❌ {name}: ERROR ({str(e)}){Style.RESET_ALL}")
        return False

def main():
    print("\n" + "="*60)
    print("🔍 ATS System Health Check")
    print("="*60 + "\n")
    
    services = [
        ("Python API", "http://localhost:8000/health"),
        ("Node.js Backend", "http://localhost:5000/health"),
        ("React Frontend", "http://localhost:5173"),
    ]
    
    results = []
    for name, url in services:
        results.append(check_service(name, url))
    
    print("\n" + "="*60)
    
    if all(results):
        print(f"{Fore.GREEN}🎉 All services are running!{Style.RESET_ALL}")
        print(f"\n{Fore.CYAN}Next steps:{Style.RESET_ALL}")
        print(f"  1. Open browser: {Fore.BLUE}http://localhost:5173{Style.RESET_ALL}")
        print(f"  2. Login as candidate and apply to a job")
        print(f"  3. Upload a PDF resume")
        print(f"  4. Check 'My Applications' for ATS score")
        print(f"\n{Fore.CYAN}Or re-analyze old applications:{Style.RESET_ALL}")
        print(f"  - Go to: {Fore.BLUE}http://localhost:5173/admin/migrate{Style.RESET_ALL}")
        print(f"  - Click 'Start Re-analysis'\n")
        sys.exit(0)
    else:
        print(f"{Fore.RED}⚠️  Some services are not running!{Style.RESET_ALL}")
        print(f"\n{Fore.YELLOW}To start missing services:{Style.RESET_ALL}")
        
        if not results[0]:  # Python API
            print(f"\n{Fore.YELLOW}Python API:{Style.RESET_ALL}")
            print(f"  cd C:\\Users\\Mahmoud\\Desktop\\integ")
            print(f"  .\\venv\\Scripts\\Activate.ps1")
            print(f"  python ats_api_service.py")
        
        if not results[1]:  # Node.js
            print(f"\n{Fore.YELLOW}Node.js Backend:{Style.RESET_ALL}")
            print(f"  cd C:\\Users\\Mahmoud\\Desktop\\integ\\server")
            print(f"  npm run dev")
        
        if not results[2]:  # React
            print(f"\n{Fore.YELLOW}React Frontend:{Style.RESET_ALL}")
            print(f"  cd C:\\Users\\Mahmoud\\Desktop\\integ\\client")
            print(f"  npm run dev")
        
        print()
        sys.exit(1)

if __name__ == "__main__":
    main()
