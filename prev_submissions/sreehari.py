import requests
import pandas as pd
import time
import random
import os
from bs4 import BeautifulSoup
from colorama import Fore, Style, init
from openpyxl import load_workbook

init(autoreset=True)

user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/108.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
]

url = "https://internshala.com/jobs/"

def get_headers():
    """Returns a dictionary of headers with a random User-Agent."""
    return {
        'Accept': 'application/x-clarity-gzip',
        'Accept-Encoding': 'gzip, deflate, br, zstd',
        'Accept-Language': 'en-US,en;q=0.9,ml;q=0.8,ja;q=0.7',
        'User-Agent': random.choice(user_agents)
    }

def scrape_jobs():
    """
    Main function to scrape job data from Internshala.
    Provides real-time progress updates and prints a summary.
    """
    print(f"{Fore.CYAN}---------------------------------------------")
    print(f"{Fore.CYAN} Starting Internshala Job Scraper")
    print(f"{Fore.CYAN}---------------------------------------------")
    print(f"{Fore.YELLOW}Fetching main job listing page...")

    try:
        response = requests.get(url, headers=get_headers())
        response.raise_for_status() 
        soup = BeautifulSoup(response.text, "html.parser")
    except requests.exceptions.RequestException as e:
        print(f"{Fore.RED}Error fetching the main page: {e}")
        return

    cards = soup.find_all("div", class_="internship_meta experience_meta")
    if not cards:
        print(f"{Fore.RED}No job cards found. The HTML structure might have changed.")
        return

    jobs = []
    print(f"{Fore.GREEN}Found {len(cards)} potential job listings. Beginning detailed scrape.")

    for i, card in enumerate(cards):
        try:
            job_name = card.find("a", class_="job-title-href").text.strip()
            job_url = "https://internshala.com" + card.find("a", class_="job-title-href")["href"]
            print(f"{Fore.BLUE}  Scraping job {i + 1}/{len(cards)}: {job_name}")
        except AttributeError:
            print(f"{Fore.RED}  Job card {i + 1} skipped due to missing title or URL.")
            continue

        location, Exp, skills, salary, about = "", "", "", "", ""

        try:
            detail_resp = requests.get(job_url, headers=get_headers())
            detail_resp.raise_for_status()
            detail_soup = BeautifulSoup(detail_resp.text, "html.parser")

            location_element = detail_soup.find("p", id="location_names")
            if location_element:
                location = location_element.find("a").text.strip() if location_element.find("a") else location_element.text.strip()

            experience_element = detail_soup.find("div", class_="job-experience-item")
            if experience_element:
                Exp = experience_element.find("div", class_="item_body").text.strip()

            skills_div = detail_soup.find("div", class_="round_tabs_container")
            if skills_div:
                skill_spans = skills_div.find_all("span", class_="round_tabs")
                skills = ", ".join([s.text.strip() for s in skill_spans])

            internship_details = detail_soup.find("div", class_="internship_details")
            if internship_details:
                text_container = internship_details.find("div", class_="text-container")
                if text_container:
                    lines = [line.strip() for line in text_container.text.split('\n') if line.strip()]
                    about = lines[0:10]

            salary_container = detail_soup.find("div", class_="text-container salary_container")
            if salary_container:
                salary = salary_container.p.text.strip()

        except requests.exceptions.RequestException as e:
            print(f"{Fore.RED}  Error fetching details from {job_url}: {e}")
            continue 

        jobs.append({
            "JobTitle": job_name,
            "Location": location,
            "Experience": Exp,
            "Skills": skills,
            "Salary": salary,
            "JobUrl": job_url,
            "JobDescriptionSummary": about
        })

        delay = random.uniform(1, 3)
        print(f"{Fore.MAGENTA}  Pausing for {delay:.2f} seconds...")
        time.sleep(delay)

    df = pd.DataFrame(jobs)

    if not df.empty:
        file = "Jobs.xlsx"
        df.to_excel(file, index=False, engine="openpyxl")
        time.sleep(1)
        try:
            wb = load_workbook(file)
            sheet = wb.active
            sheet.title = "Jobs"
            print(f"{Fore.GREEN}Jobs data saved to '{file}'. Formatting Excel...")

            for column in sheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except (TypeError, ValueError):
                        pass
                adjusted_width = (max_length + 2)
                sheet.column_dimensions[column_letter].width = adjusted_width

            wb.save(file)
            print(f"{Fore.GREEN}Excel formatting complete.")

            print(f"{Fore.CYAN}---------------------------------------------")
            print(f"{Fore.CYAN} Scraper Summary")
            print(f"{Fore.CYAN}---------------------------------------------")
            print(f"{Fore.GREEN}Total jobs scraped: {len(jobs)}")
            print(f"{Fore.GREEN}Total jobs saved to Excel: {len(df)}")
            print("\n" + Fore.YELLOW + "Here's a preview of the first 5 jobs:")
            print(df.head().to_string())
            print(f"{Fore.CYAN}---------------------------------------------")

        except Exception as e:
            print(f"{Fore.RED}An error occurred while formatting the Excel file.")
            print(f"{Fore.RED}Please make sure the file '{file}' is not open in another program and try again.")
            print(f"{Fore.RED}Error details: {e}")
    else:
        print(f"{Fore.RED}No job data was scraped. Exiting.")

if __name__ == "__main__":
    scrape_jobs()
