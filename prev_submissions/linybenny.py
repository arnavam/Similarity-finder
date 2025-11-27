import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin
import time

def scrape_job_details(job_url, headers):
    """Fetch job description summary and skills from the detailed job page."""
    try:
        response = requests.get(job_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')

        description_summary = "N/A"
        skills = []

        blog_section = soup.find('section', id='blog')
        if blog_section:
            text_blocks = blog_section.find_all(['p', 'h2', 'h3', 'li'])

            #  JobDescriptionSummary
            for i, block in enumerate(text_blocks):
                if "Role Overview" in block.get_text():
                    if i + 1 < len(text_blocks):
                        description_summary = text_blocks[i + 1].get_text(strip=True)
                    break

            #  Skills 
            found_section = None
            for i, block in enumerate(text_blocks):
                if "Requirements" in block.get_text() or "skills" in block.get_text().lower():
                    found_section = i
                    break
                

            if found_section is not None:
                for j in range(found_section + 1, len(text_blocks)):
                    if text_blocks[j].name == "li":
                        skills.append(text_blocks[j].get_text(strip=True))
                    elif text_blocks[j].name in ["h2", "h3"]:
                        break

        return description_summary, ', '.join(skills[:7]) if skills else 'N/A'
    except:
        return "N/A", "N/A"


def scrape_techvantage_jobs():
    """Scrape job postings from TechVantage careers page and export to Excel."""
    base_url = "https://www.techvantagesystems.com/careers/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    all_jobs = []
    response = requests.get(base_url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    job_listings = soup.find_all('div', class_='job-listing')

    for job in job_listings:
        job_data = {
            'JobTitle': 'N/A',
            'Location': 'N/A',
            'ExperienceRequired': 'N/A',
            'SkillsRequired': 'N/A',
            'Salary': 'N/A',
            'JobURL': 'N/A',
            'JobDescriptionSummary': 'N/A'
        }

        # Job title and URL
        title_elem = job.find('h4', class_='job__title')
        if title_elem and title_elem.find('a'):
            job_data['JobTitle'] = title_elem.find('a').text.strip()
            job_data['JobURL'] = urljoin(base_url, title_elem.find('a')['href'])

        #  Experience and Location
        text_parts = job.get_text(separator="|").split("|")
        for part in text_parts:
            if "years" in part.lower():
                job_data['ExperienceRequired'] = part.strip()
            if "Trivandrum" in part or "Kerala" in part:
                job_data['Location'] = part.strip()

        
        if job_data['JobURL'] != 'N/A':
            desc_summary, skills = scrape_job_details(job_data['JobURL'], headers)
            job_data['JobDescriptionSummary'] = desc_summary
            job_data['SkillsRequired'] = skills
            time.sleep(1)

        all_jobs.append(job_data)

    
    df = pd.DataFrame(all_jobs, columns=[
        'JobTitle',
        'Location',
        'ExperienceRequired',
        'SkillsRequired',
        'Salary',
        'JobURL',
        'JobDescriptionSummary'
    ])
    df.to_excel('Techvantage_Jobs.xlsx', index=False, engine='openpyxl')

    print(f"Scraped {len(all_jobs)} jobs.")
    return all_jobs


if __name__ == "__main__":
    scrape_techvantage_jobs()
