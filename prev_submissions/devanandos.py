"""
Job Scraper for RemoteOK (Python/Remote Jobs)
Author: Your Name
Description: Scrapes job listings and details from RemoteOK and saves them into an Excel file.
"""

import re

import pandas as pd
import requests
from bs4 import BeautifulSoup

# -------------------------
# CONFIG
# -------------------------
START_URL = "https://remoteok.com/remote-dev+python-jobs.json"
HEADERS = {"User-Agent": "Mozilla/5.0"}
EXCEL_FILENAME = "RemoteOK_Jobs_Full.xlsx"

# -------------------------
# Fetch job list from JSON
# -------------------------
data = requests.get(START_URL, headers=HEADERS).json()

jobs = []

for item in data:
    if item.get("slug"):  # ensures it's a job post
        title = item.get("position")
        company = item.get("company")
        location = item.get("location") or "Not Specified"

        job_url = item.get("url")
        if not job_url.startswith("http"):
            job_url = "https://remoteok.com" + job_url

        jobs.append([title, company, location, job_url])

# -------------------------
# Convert to DataFrame
# -------------------------
df = pd.DataFrame(jobs, columns=["Title", "Company", "Location", "URL"])

# -------------------------
# Function to extract job details
# -------------------------


def extract_job_details(url):
    try:
        r = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(r.text, "html.parser")

        # Description
        desc_section = soup.find("div", {"class": "description"})
        desc = desc_section.get_text(" ", strip=True) if desc_section else ""

        # Experience
        exp_pattern = r"(\d+\+?\s+years?)"
        experience = ", ".join(re.findall(exp_pattern, desc, re.IGNORECASE))

        # Salary
        salary_pattern = r"(\$\s?\d[\d,\.]*\s?(?:-\s?\$\s?\d[\d,\.]*)?)"
        salary = ", ".join(re.findall(salary_pattern, desc))

        # Skills
        skills_list = [
            "Python",
            "JavaScript",
            "React",
            "Node",
            "AWS",
            "GCP",
            "Azure",
            "Docker",
            "Kubernetes",
            "Linux",
            "SQL",
            "NoSQL",
            "Django",
            "Flask",
            "Terraform",
            "CI/CD",
            "Machine Learning",
        ]
        skills = ", ".join(
            [s for s in skills_list if s.lower() in desc.lower()])

        # Summary
        summary = desc[:200] + "..." if len(desc) > 200 else desc

        return experience, skills, salary, summary

    except Exception as e:
        print("Error scraping:", url, e)
        return "", "", "", ""
        """
        Initialize with customizable preprocessing options

        Args:
            preprocessing_options (dict): Options for preprocessing
                - remove_comments (bool): Whether to remove comments
                - normalize_whitespace (bool): Whether to normalize whitespace
                - preserve_variable_names (bool): Whether to keep variable names in AST
                - preserve_literals (bool): Whether to keep literals in AST
        """

# -------------------------
# Add detailed columns
# -------------------------
df["ExperienceRequired"] = ""
df["SkillsRequired"] = ""
df["Salary"] = ""
df["JobDescriptionSummary"] = ""

for i, row in df.iterrows():
    print(f"Scraping job {i + 1}/{len(df)} → {row['URL']}")
    exp, skills, sal, summary = extract_job_details(row["URL"])
    df.loc[i, "ExperienceRequired"] = exp
    df.loc[i, "SkillsRequired"] = skills
    df.loc[i, "Salary"] = sal
    df.loc[i, "JobDescriptionSummary"] = summary

# -------------------------
# Save final Excel
# -------------------------
df.to_excel(EXCEL_FILENAME, index=False)
print(f"\n🎉 Done! Saved → {EXCEL_FILENAME}")
