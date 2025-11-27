import requests

from bs4 import BeautifulSoup

import pandas as pd

import time

def scrape_jobs(careers_url, company_name):
    jobs = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    try:
        response = requests.get(careers_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        job_selectors = [
            '.job-listing', '.job-item', '.careers-item', '.position',
            '.job', '.listing-item', 'li.job', 'div.job'
        ]
        
        job_elements = []
        for selector in job_selectors:
            elements = soup.select(selector)
            if elements:
                job_elements = elements
                print(f"Found {len(job_elements)} jobs using: {selector}")
                break
        
        if not job_elements:
            all_elements = soup.find_all(['div', 'li', 'tr', 'section'])
            job_elements = [elem for elem in all_elements if 
                           elem.get_text(strip=True) and 
                           len(elem.get_text(strip=True)) > 20 and
                           len(elem.get_text(strip=True)) < 500]
            print(f"Found {len(job_elements)} potential job elements")
        
        for job_elem in job_elements:
            job_data = extract_job_details(job_elem, careers_url)
            if job_data and job_data.get('title'):
                jobs.append(job_data)
                print(f"✓ {job_data['title']}")
        
        if jobs:
            save_to_excel(jobs, company_name)
            print(f"\n✅ Success! Saved {len(jobs)} jobs to {company_name}_jobs.xlsx")
        else:
            print("\n❌ No jobs found. Try adjusting the selectors.")
            
    except Exception as e:
        print(f"Error: {e}")

def extract_job_details(job_elem, base_url):
    job = {
        'title': '',
        'location': '',
        'department': '',
        'type': '',
        'url': '',
        'description': ''
    }
    
    try:
        title_selectors = ['h1', 'h2', 'h3', 'h4', '.title', '.job-title', 'a']
        for selector in title_selectors:
            title_elem = job_elem.find(selector)
            if title_elem and title_elem.get_text(strip=True):
                job['title'] = title_elem.get_text(strip=True)
                break
        
        link_elem = job_elem.find('a')
        if link_elem and link_elem.get('href'):
            href = link_elem.get('href')
            job['url'] = href if href.startswith('http') else f"{base_url}{href}"
        
        full_text = job_elem.get_text(separator=' | ', strip=True)
        
        location_keywords = ['location', 'city', 'country', 'remote', 'hybrid', 'office']
        lines = full_text.split('|')
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in location_keywords):
                job['location'] = line.strip()
                break
        
        type_keywords = ['full-time', 'part-time', 'contract', 'freelance', 'internship']
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in type_keywords):
                job['type'] = line.strip()
                break
        
        job['description'] = full_text[:100] + '...' if len(full_text) > 100 else full_text
        
    except Exception as e:
        print(f"Error extracting job details: {e}")
    
    return job

def save_to_excel(jobs, company_name):
    df = pd.DataFrame(jobs)
    
    columns_order = ['title', 'location', 'type', 'department', 'url', 'description']
    existing_columns = [col for col in columns_order if col in df.columns]
    df = df[existing_columns + [col for col in df.columns if col not in columns_order]]
    
    filename = f"{company_name}_jobs.xlsx"
    df.to_excel(filename, index=False, engine='openpyxl')
    
    print(f"\n📊 Excel file '{filename}' created successfully!")

def main():
    print("🚀 Simple Job Scraper")
    print("=" * 40)
    
    company_name = input("Enter company name: ").strip()
    careers_url = input("Enter careers page URL: ").strip()
    
    if not company_name or not careers_url:
        print("❌ Please provide both company name and URL")
        return
    
    print(f"\n🕐 Scraping jobs from {careers_url}...")
    
    scrape_jobs(careers_url, company_name)

if __name__ == "__main__":
    main()
